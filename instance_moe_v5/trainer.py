"""Stage-two MGD with instance-region experts and train-only query supervision."""

import hashlib
import json
from pathlib import Path

import torch

import moe_kd_v3.trainer as core
from moe_kd_v3.lookahead import marginal_gain, virtual_loss, virtual_loss_and_gradient
from hierarchical_moe.trainer import HierarchicalDistiller, addon_hash as hierarchical_hash

from .loss import InstanceFeatureLoss


EXPERTS = ("cwd", "mimic")


def addon_hash():
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    digest.update(hierarchical_hash().encode())
    for name in ("instance_moe_v5/loss.py", "instance_moe_v5/trainer.py", "train_instance_moe_v5.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class InstanceDistiller(HierarchicalDistiller):
    """Restore the same MGD student and auxiliary states as previous controls."""

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        # Inherited state saves and restores resolve this function at runtime.
        core.addon_hash = addon_hash

    def build_optimizer(self, model, **kwargs):
        old = kwargs.pop("feature_model")
        teacher_channels = [layer.num_features for layer in old.norm]
        student_channels = [
            layer.in_channels if isinstance(layer, torch.nn.Conv2d) else target
            for layer, target in zip(old.align_module, teacher_channels)
        ]
        if student_channels != [64, 128, 256] or teacher_channels != [128, 256, 512]:
            raise ValueError(f"Expected YOLOv8n/s feature channels: {student_channels}, {teacher_channels}")
        module = InstanceFeatureLoss(student_channels, teacher_channels, self.anchor_config).to(self.device)
        self.kd_feature_loss = module
        if core.file_hash(self.stage1_aux_path) != self.anchor_config.stage1_aux_sha256:
            raise ValueError("Stage-one MGD auxiliary checkpoint changed")
        state = torch.load(self.stage1_aux_path, map_location="cpu", weights_only=False)
        config = state.get("config", {})
        if (config.get("mode") != "mgd" or config.get("mgd_alpha") != self.anchor_config.mgd_alpha
                or config.get("mask_drop") != self.anchor_config.mask_drop):
            raise ValueError("Stage-one MGD checkpoint mode, alpha or mask differs")
        auxiliary = state.get("auxiliary")
        if not isinstance(auxiliary, dict):
            raise ValueError("Stage-one checkpoint has no auxiliary state")
        mgd = {key[4:]: value for key, value in auxiliary.items() if key.startswith("mgd.")}
        if not mgd:
            raise ValueError("Stage-one checkpoint has no MGD state")
        module.mgd.load_state_dict(mgd, strict=True)
        for name, value in module.mgd.state_dict().items():
            if not torch.equal(value.detach().cpu(), mgd[name].detach().cpu()):
                raise RuntimeError(f"Stage-one MGD auxiliary mismatch: {name}")
        del state, auxiliary, mgd
        print("STAGE1 MGD AUXILIARY EXACT LOAD PASS", flush=True)
        optimizer = super(core.AnchorDistiller, self).build_optimizer(model, feature_model=module, **kwargs)
        gate = list(module.gate.parameters())
        ids = {id(parameter) for parameter in gate}
        for group in optimizer.param_groups:
            group["params"] = [parameter for parameter in group["params"] if id(parameter) not in ids]
        optimizer.add_param_group({"params": gate, "lr": kwargs["lr"] * self.anchor_config.gate_lr_scale,
                                   "weight_decay": 0.0})
        return optimizer

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        path = self.save_dir / "anchor_protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        protocol.update({
            "version": "instance_moe_v5",
            "algorithm": self.kd_feature_loss.metadata(),
            "query_source": "previous augmented training batch only",
            "gate_target_data": "current augmented training instance and previous augmented training query",
            "gate_target": "signed first-order query gains normalized by maximum accepted absolute gain, "
                           "divided by gate_target_temperature and softmax over accepted experts plus none",
            "virtual_update": "read-only functional plain SGD surrogate; no momentum or accumulation",
            "gain_note": "proxy gain is not AP and is not an exact optimizer replay",
            "norm_budget": "per-scale and per-expert EMA of feature-gradient norm ratios; not a strict bound",
            "stage1_weight_pair": "student is best-epoch EMA; MGD module is the same epoch raw training state",
            "route_logging": "training-time augmented GT instances, not inference-time routing",
        })
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    def _task_supervise(self, features, teacher_features, base_vectors, values, probs, instances):
        """Train-only query supervision; do not mutate live weights, BN, RNG or grads."""
        module, config = self.kd_feature_loss, self.anchor_config
        zero = probs.new_zeros(())
        if not instances or self.batch_index % config.probe_every:
            return zero
        valid = module.valid_experts
        if valid.shape != values.shape or probs.shape != (len(instances), 3):
            raise ValueError("Invalid instance routing tensors")
        scales = torch.tensor([item[3] for item in instances], device=values.device)
        norm_rows = []
        for level, feature in enumerate(features):
            level_mask = scales == level
            if not level_mask.any():
                continue
            base = torch.autograd.grad(base_vectors[level].mean(), feature, retain_graph=True,
                                       allow_unused=True)[0]
            if base is None:
                continue
            base_norm = base.detach().float().norm()
            if not torch.isfinite(base_norm):
                raise FloatingPointError("Invalid MGD feature gradient")
            if base_norm.item() <= 1e-12:
                continue
            for expert, name in enumerate(EXPERTS):
                selected = level_mask & valid[:, expert]
                if not selected.any():
                    continue
                gradient = torch.autograd.grad(values[selected, expert].mean(), feature,
                                               retain_graph=True, allow_unused=True)[0]
                if gradient is None:
                    continue
                norm = gradient.detach().float().norm()
                if not torch.isfinite(norm):
                    raise FloatingPointError("Invalid expert feature gradient")
                if norm.item() <= 1e-12:
                    continue
                desired = (base_norm / norm).clamp(1e-8, config.max_scale)
                with torch.no_grad():
                    if not module.scale_updates[level, expert].item():
                        module.aux_scales[level, expert].copy_(desired)
                    else:
                        module.aux_scales[level, expert].lerp_(desired, config.scale_ema)
                    module.scale_updates[level, expert].add_(1)
                norm_rows.append({"epoch": self.epoch + 1, "batch": self.batch_index,
                                  "scale": f"P{level + 3}", "expert": name,
                                  "instances": int(selected.sum().item()), "mgd_norm": float(base_norm),
                                  "aux_norm": float(norm), "factor": float(module.aux_scales[level, expert])})
        core.append_csv(self.save_dir / "norm_calibration.csv", norm_rows)
        if self.query_buffer is None or not config.task_coef:
            return zero
        if self.capture is None or self.capture.last is None:
            raise RuntimeError("Detection objective is unavailable")
        params = [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        base_loss = self.capture.last + sum(value.mean() for value in base_vectors) * self.args.feature_loss_ratio
        gradients = torch.autograd.grad(base_loss, [value for _, value in params], retain_graph=True,
                                        allow_unused=True)
        base_grads = {name: grad.detach() if grad is not None else None
                      for (name, _), grad in zip(params, gradients)}
        query = self.query_buffer
        base_query, query_grads = virtual_loss_and_gradient(
            self.model, self.query_criterion, query, base_grads, step=config.virtual_lr
        )
        count = min(config.probe_samples, len(instances))
        offset = self.epoch * 17 + (self.batch_index // config.probe_every) * config.probe_samples
        indices = [(offset + j) % len(instances) for j in range(count)]
        objective, rows = [], []
        for index in indices:
            sample, class_id, box, level, _ = instances[index]
            instance_count = sum(item[0] == sample for item in instances)
            image_count = features[0].shape[0]
            available = [bool(valid[index, e].item() and module.scale_updates[level, e].item())
                         for e in range(len(EXPERTS))]
            # A masked expert does not participate in the live instance objective.
            if not any(available):
                continue
            result = {}
            accepted = [False, False]
            estimates = [0.0, 0.0]
            for expert, name in enumerate(EXPERTS):
                if not available[expert]:
                    continue
                item = (values[index, expert] * module.aux_scales[level, expert].detach()
                        * config.aux_fraction * self.args.feature_loss_ratio / (image_count * instance_count))
                gradients = torch.autograd.grad(item, [value for _, value in params],
                                                retain_graph=True, allow_unused=True)
                aux_grads = {key: grad.detach() if grad is not None else None
                             for (key, _), grad in zip(params, gradients)}
                actual = virtual_loss(self.model, self.query_criterion, query, base_grads, aux_grads,
                                      step=config.virtual_lr)
                observed = (base_query - actual) / max(abs(base_query), 1e-8)
                estimate, alignment = marginal_gain(base_grads, aux_grads, query_grads, base_query,
                                                    config.virtual_lr)
                tolerance = max(5e-7, abs(estimate) * 0.5)
                sign_agrees = estimate * observed >= 0 or abs(observed) <= tolerance
                accepted[expert] = (abs(estimate) > config.min_gain
                                    and abs(alignment) > config.min_alignment and sign_agrees)
                estimates[expert] = estimate
                result.update({f"{name}_observed": observed, f"{name}_estimated": estimate,
                               f"{name}_alignment": alignment, f"{name}_accepted": int(accepted[expert])})
            # Skip weak/contradictory probes rather than forcing an artificial none label.
            if any(accepted):
                magnitude = max(abs(estimates[e]) for e in range(2) if accepted[e])
                logits = probs.new_tensor([0.0] + [
                    estimates[e] / magnitude if accepted[e] else -1e4 for e in range(2)
                ])
                target = (logits / config.gate_target_temperature).softmax(0).detach()
                # A weak or contradicted expert carries no label, so it must
                # not be implicitly penalized by this probe's cross-entropy.
                mask = probs.new_tensor([1.0, float(accepted[0]), float(accepted[1])])
                normalized = probs[index] * mask
                normalized = normalized / normalized.sum().clamp_min(1e-12)
                objective.append(-(target * normalized.clamp_min(1e-12).log()).sum())
            else:
                target = None
            row = {"epoch": self.epoch + 1, "batch": self.batch_index, "sample": int(sample),
                   "class_id": int(class_id), "scale": f"P{level + 3}",
                   "query_files": json.dumps(query["im_file"], ensure_ascii=False),
                   "base_query_loss": base_query}
            for name in EXPERTS:
                for suffix in ("observed", "estimated", "alignment", "accepted"):
                    row[f"{name}_{suffix}"] = result.get(f"{name}_{suffix}", "")
            row.update({"target_none": float(target[0]) if target is not None else "",
                        "target_cwd": float(target[1]) if target is not None else "",
                        "target_mimic": float(target[2]) if target is not None else "",
                        "trained": int(target is not None)})
            rows.append(row)
        core.append_csv(self.save_dir / "task_probes.csv", rows)
        return torch.stack(objective).mean() if objective else zero

    def _diagnose(self, trace):
        # The V3 observer expects a different trace schema.
        pass

    def _batch_end(self, trainer):
        module, trace = self.kd_feature_loss, self.kd_feature_loss.trace
        core.append_csv(self.save_dir / "instance_losses.csv",
                        [{"epoch": self.epoch + 1, "batch": self.batch_index, **module.last_stats}])
        if self.batch_index % 50 == 0:
            instances = trace.get("instances", [])
            probs = trace.get("route")
            if probs is not None and len(instances):
                by_group = {}
                for instance, values in zip(instances, probs.detach().cpu().tolist()):
                    _, class_id, _, level, _ = instance
                    by_group.setdefault((class_id, level), []).append(values)
                rows = []
                for (class_id, level), group in sorted(by_group.items()):
                    rows.append({"epoch": self.epoch + 1, "batch": self.batch_index,
                                 "class_id": class_id, "scale": f"P{level + 3}",
                                 "instances": len(group),
                                 "p_none": sum(x[0] for x in group) / len(group),
                                 "p_cwd": sum(x[1] for x in group) / len(group),
                                 "p_mimic": sum(x[2] for x in group) / len(group)})
                core.append_csv(self.save_dir / "instance_routes.csv", rows)
            print(f"\n[instance v5 epoch={self.epoch + 1} batch={self.batch_index}] {module.last_stats}", flush=True)
        self.query_buffer = core.query_slice(self.current_batch, self.anchor_config.probe_samples)
        module.trace = {}
        module.last_scale_losses = ()
        module.batch = None
        self.capture.last = None
