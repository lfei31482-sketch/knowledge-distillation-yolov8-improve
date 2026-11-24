"""MGD with calibrated per-object auxiliary CWD and Mimic experts."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe_kd_v3.loss import AnchorConfig, Aligner, RawMGD, auxiliary_vector


EXPERTS = ("cwd", "mimic")
CHOICES = ("none", *EXPERTS)
SCALES = ("P3", "P4", "P5")


@dataclass
class InstanceConfig(AnchorConfig):
    method_version: str = "instance_moe_v5"
    mode: str = "gated"
    warmup_epochs: int = 20
    ramp_epochs: int = 10
    aux_fraction: float = 0.1
    task_coef: float = 0.1
    gate_temperature: float = 1.0
    gate_target_temperature: float = 1.0
    stage1_path: str = ""
    stage1_sha256: str = ""
    stage1_aux_sha256: str = ""
    learning_rate: float = 0.001

    def validate(self):
        super().validate()
        if self.method_version != "instance_moe_v5":
            raise ValueError("Invalid instance MoE method version")
        if self.mode != "gated":
            raise ValueError("Only instance-gated mode is supported")
        if min(self.gate_temperature, self.gate_target_temperature, self.learning_rate) <= 0:
            raise ValueError("Gate temperatures and learning rate must be positive")
        if bool(self.stage1_path) != bool(self.stage1_sha256):
            raise ValueError("Stage-one path and SHA-256 must be supplied together")
        for value in (self.stage1_sha256, self.stage1_aux_sha256):
            if value and len(value) != 64:
                raise ValueError("Invalid SHA-256")


def collect_instances(batch, features, classes=20):
    """Extract nonempty augmented-image boxes and their clipped feature-grid cells.

    Exactly one level is assigned by the square root of area in input pixels,
    with 64 and 160 pixel boundaries. This is a routing heuristic.
    """
    if batch is None:
        raise ValueError("Augmented training batch is required")
    ids = batch["batch_idx"].detach().reshape(-1).cpu().tolist()
    labels = batch["cls"].detach().reshape(-1).cpu().tolist()
    boxes = batch["bboxes"].detach().reshape(-1, 4).cpu().tolist()
    if len(ids) != len(labels) or len(ids) != len(boxes):
        raise ValueError("Mismatched batch indices, classes, and boxes")
    image_h, image_w = batch["img"].shape[-2:]
    result = []
    for image, label, coords in zip(ids, labels, boxes):
        if not all(math.isfinite(float(x)) for x in (image, label, *coords)):
            raise ValueError("Non-finite training target")
        image_index, class_id = int(image), int(label)
        if image != image_index or label != class_id:
            raise ValueError("Image index and class id must be integers")
        if not (0 <= image_index < features[0].shape[0] and 0 <= class_id < classes):
            raise ValueError("Image index or class id outside range")
        cx, cy, width, height = map(float, coords)
        if width <= 0 or height <= 0:
            continue
        left, right = max(0., cx - width / 2), min(1., cx + width / 2)
        top, bottom = max(0., cy - height / 2), min(1., cy + height / 2)
        if left >= right or top >= bottom:
            continue
        # Use the visible, clipped box for both geometry and region extraction.
        clipped = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
        size = math.sqrt(clipped[2] * clipped[3] * image_h * image_w)
        level = int(size >= 64) + int(size >= 160)
        h, w = features[level].shape[-2:]
        xyxy = (
            min(w - 1, max(0, math.floor(left * w))),
            min(h - 1, max(0, math.floor(top * h))),
            min(w, max(1, math.ceil(right * w))),
            min(h, max(1, math.ceil(bottom * h))),
        )
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        result.append((image_index, class_id, clipped, level, xyxy))
    return result


class InstanceGate(nn.Module):
    """Three choices per annotated object, using detached student ROI features."""

    def __init__(self, channels, hidden, temperature):
        super().__init__()
        self.temperature = temperature
        self.project = nn.ModuleList([nn.Linear(channels_at_scale, 32) for channels_at_scale in channels])
        self.embedding = nn.Embedding(20, 16)
        self.net = nn.Sequential(nn.Linear(32 + 16 + 2, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, len(CHOICES)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, instances, valid_experts):
        if not instances:
            return features[0].new_empty((0, len(CHOICES)))
        values = []
        for image_index, class_id, box, level, (x1, y1, x2, y2) in instances:
            pooled = features[level][image_index, :, y1:y2, x1:x2].detach().float().mean((1, 2))
            geometry = pooled.new_tensor([math.log(max(box[2] * box[3], 1e-8)),
                                          math.log(max(box[2], 1e-8) / max(box[3], 1e-8))])
            value = torch.cat((self.project[level](pooled), self.embedding.weight[class_id], geometry))
            values.append(self.net(value))
        logits = torch.stack(values) / self.temperature
        choices_valid = torch.cat((torch.ones((len(instances), 1), dtype=torch.bool, device=logits.device),
                                   valid_experts), dim=1)
        logits = logits.masked_fill(~choices_valid, torch.finfo(logits.dtype).min)
        return logits.softmax(dim=1)


class InstanceFeatureLoss(nn.Module):
    """One full MGD base plus per-instance bounded auxiliary contributions."""

    def __init__(self, student_channels, teacher_channels, config=None):
        super().__init__()
        self.config = config or InstanceConfig()
        self.config.validate()
        self.mgd = RawMGD(student_channels, teacher_channels, self.config.mgd_alpha, self.config.mask_drop)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed + 7907)
            self.adapters = nn.ModuleDict({expert: Aligner(student_channels, teacher_channels) for expert in EXPERTS})
            self.gate = InstanceGate(student_channels, self.config.hidden, self.config.gate_temperature)
        self.register_buffer("aux_scales", torch.ones(3, len(EXPERTS)))
        self.register_buffer("scale_updates", torch.zeros(3, len(EXPERTS), dtype=torch.long))
        self.epoch = 0
        self.batch = None
        self.task_supervisor = None
        self.observer = None
        self.last_scale_losses = ()
        self.last_stats = {}
        self.trace = {}
        self.valid_experts = None

    @property
    def feature_loss(self):
        return self

    def router_state(self):
        return None

    def set_router_state(self, value):
        if value is not None:
            raise ValueError("Soft gate has no separate sampler RNG state")

    def metadata(self):
        return {
            "config": asdict(self.config), "choices": CHOICES, "scales": SCALES,
            "mgd_coefficient": 1.0, "mgd": "full original RawMGD at P3/P4/P5",
            "routing_unit": "augmented GT instance, assigned one heuristic scale",
            "scale_thresholds_pixels": [64, 160],
            "gate_inputs": "detached student ROI feature, GT class, clipped area and aspect ratio",
            "cwd": "per-channel spatial KL within the clipped grid ROI; disabled for fewer than two cells",
            "mimic": "mean squared error within the clipped grid ROI",
            "overlaps": "separate instances; each annotated image has equal total auxiliary weight",
            "uncalibrated": "experts are masked until their per-scale gradient calibration exists",
            "no_targets": "zero auxiliary loss; full MGD remains",
            "gate_gradient": "separate task supervision, detached route in student loss",
        }

    def fraction(self):
        if self.epoch < self.config.warmup_epochs:
            return 0.0
        ramp = min((self.epoch - self.config.warmup_epochs + 1) / max(self.config.ramp_epochs, 1), 1.)
        return self.config.aux_fraction * ramp

    def instance_values(self, features, teachers, instances):
        """Return expert costs and geometry validity in instance order."""
        aligned = {name: self.adapters[name](features, teachers) for name in EXPERTS}
        values, valid = [], []
        for image_index, _, _, level, (x1, y1, x2, y2) in instances:
            s, t = aligned["cwd"][level]
            a = s[image_index, :, y1:y2, x1:x2].flatten(1)
            b = t[image_index, :, y1:y2, x1:x2].flatten(1)
            enough_cells = a.shape[1] >= 2
            if enough_cells:
                log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
                cwd = (log_b.exp() * (log_b - log_a)).sum(-1).mean()
            else:
                # A one-cell softmax is identically one. Do not route to CWD.
                cwd = a.sum() * 0.
            s, t = aligned["mimic"][level]
            mimic = (s[image_index, :, y1:y2, x1:x2] - t[image_index, :, y1:y2, x1:x2]).square().mean()
            values.append(torch.stack((cwd, mimic)))
            valid.append((enough_cells, True))
        if not values:
            return features[0].new_empty((0, len(EXPERTS))), torch.empty(
                (0, len(EXPERTS)), device=features[0].device, dtype=torch.bool
            )
        return torch.stack(values), torch.tensor(valid, device=features[0].device, dtype=torch.bool)

    def forward(self, student_features, teacher_features, batch=None):
        if len(student_features) != 3 or len(teacher_features) != 3:
            raise ValueError("Expected P3, P4, P5 features")
        current_batch = self.batch if batch is None else batch
        with torch.autocast(device_type=student_features[0].device.type, enabled=False):
            sf = [x.float() for x in student_features]
            tf = [x.detach().float() for x in teacher_features]
            vectors = self.mgd(sf, tf)
            if not all(torch.isfinite(vector).all() for vector in vectors):
                raise FloatingPointError("Non-finite MGD")
            base = [vector.mean() for vector in vectors]
            fraction = self.fraction()
            residual = [base[0].new_zeros(()) for _ in range(3)]
            shadow = base[0].new_zeros(())
            task = base[0].new_zeros(())
            instances = []
            values = base[0].new_empty((0, len(EXPERTS)))
            probs = base[0].new_empty((0, len(CHOICES)))
            route = probs
            self.valid_experts = torch.empty((0, len(EXPERTS)), device=sf[0].device, dtype=torch.bool)
            if fraction:
                instances = collect_instances(current_batch, sf)
                if instances:
                    values, self.valid_experts = self.instance_values(sf, tf, instances)
                    if not torch.isfinite(values).all():
                        raise FloatingPointError("Non-finite auxiliary instance loss")
                    probs = self.gate(sf, instances, self.valid_experts)
                    if self.task_supervisor is not None:
                        task = self.task_supervisor(sf, tf, vectors, values, probs, instances)
                    levels = torch.tensor([item[3] for item in instances], device=sf[0].device)
                    image_indices = torch.tensor([item[0] for item in instances], device=sf[0].device)
                    calibrated = self.scale_updates[levels] > 0
                    eligible = self.valid_experts & calibrated
                    valid_choices = torch.cat((torch.ones((len(instances), 1), device=sf[0].device,
                                                           dtype=torch.bool), eligible), dim=1)
                    route = probs.detach() * valid_choices
                    route = route / route.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    weighted = (route[:, 1:] * self.aux_scales[levels].detach() * values).sum(dim=1)
                    image_counts = torch.bincount(image_indices, minlength=sf[0].shape[0]).clamp_min(1)
                    weighted = weighted / image_counts[image_indices] / sf[0].shape[0]
                    for level in range(3):
                        residual[level] = fraction * weighted[levels == level].sum()
            elif self.config.adapter_pretrain:
                detached = [x.detach() for x in sf]
                for name in EXPERTS:
                    for s, t in self.adapters[name](detached, tf):
                        value = auxiliary_vector(name, s, t).mean()
                        shadow = shadow + value / value.detach().clamp_min(1e-12)
                shadow = self.config.adapter_pretrain * shadow
            extra = (self.config.task_coef * task + shadow) / 3
            self.last_scale_losses = tuple(b + r + extra for b, r in zip(base, residual))
            result = sum(self.last_scale_losses)
            if not torch.isfinite(result):
                raise FloatingPointError("Non-finite instance MoE loss")
            self.last_stats = {
                "mgd_raw": sum(base).detach().item(), "aux_weighted": sum(residual).detach().item(),
                "gate_task_loss": task.detach().item(), "adapter_pretrain": shadow.detach().item(),
                "aux_fraction": fraction, "instances": len(instances),
                "prob_none": route[:, 0].mean().item() if len(route) else 0.,
                "prob_cwd": route[:, 1].mean().item() if len(route) else 0.,
                "prob_mimic": route[:, 2].mean().item() if len(route) else 0.,
            }
            self.trace = {
                "features": sf, "base_vectors": vectors, "base_loss": sum(base),
                "values": values, "instances": instances, "probs": probs.detach(),
                "route": route.detach(), "valid_experts": self.valid_experts.detach(),
                "fraction": fraction, "active": bool(fraction),
            }
            if self.observer is not None:
                self.observer(self.trace)
            return result
