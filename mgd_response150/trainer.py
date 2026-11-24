"""MGD + response-only ablation using the identical single-stage engine."""
import hashlib, json
from pathlib import Path
import moe_kd_v3.trainer as core
from single_moe150.trainer import SingleDistiller, addon_hash as single_hash

def addon_hash():
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256(single_hash().encode())
    for name in ("mgd_response150/trainer.py", "train_mgd_response150.py"):
        h.update(name.encode()); h.update((root / name).read_bytes())
    return h.hexdigest()

class ResponseOnlyDistiller(SingleDistiller):
    def __init__(self, config, **kwargs):
        if config.aux_fraction != 0 or config.adapter_pretrain != 0 or config.task_coef != 0:
            raise ValueError("The ablation must disable gate and auxiliary experts")
        super().__init__(config=config, **kwargs)
        core.addon_hash = addon_hash

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        path = self.save_dir / "anchor_protocol.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update({"version": "mgd_response150", "ablation": "MGD plus response only",
                     "instance_gate_and_experts": "constructed but inactive; no forward or gradient",
                     "aux_fraction": 0, "adapter_pretrain": 0, "task_coef": 0,
                     "mode_field_explanation": "gated is a compatibility field; zero auxiliary fraction prevents gate/expert forward"})
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
