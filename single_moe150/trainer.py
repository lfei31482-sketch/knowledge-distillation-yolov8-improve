"""Single-stage trainer; never loads a previous MGD student or auxiliary state."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
import moe_kd_v3.trainer as core
from instance_moe_v5.loss import InstanceConfig, InstanceFeatureLoss
from instance_moe_v5.trainer import InstanceDistiller, addon_hash as v5_hash
from instance_moe_response_v1.response import ResponseLoss

@dataclass
class SingleConfig(InstanceConfig):
    single_stage: bool = True
    response_ratio: float = .05
    response_temperature: float = 2.
    response_conf: float = .25
    response_iou: float = .30
    response_warmup: int = 30

    def validate(self):
        super().validate()
        if not self.single_stage or self.stage1_path or self.stage1_sha256 or self.stage1_aux_sha256:
            raise ValueError('Single-stage run cannot load any stage-one state')

def addon_hash():
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256(v5_hash().encode())
    for name in ('single_moe150/trainer.py','train_single_moe150.py',
                 'instance_moe_response_v1/response.py','ultralytics/utils/loss.py',
                 'ultralytics/models/yolo/detect/distill.py'):
        h.update(name.encode()); h.update((root/name).read_bytes())
    return h.hexdigest()

class SingleDistiller(core.AnchorDistiller):
    _task_supervise = InstanceDistiller._task_supervise
    _diagnose = InstanceDistiller._diagnose
    _batch_end = InstanceDistiller._batch_end

    def __init__(self, config, **kwargs):
        config.validate()
        super().__init__(config=config, **kwargs)
        core.addon_hash = addon_hash

    def build_optimizer(self, model, **kwargs):
        old = kwargs.pop('feature_model')
        tc = [layer.num_features for layer in old.norm]
        sc = [layer.in_channels if isinstance(layer, torch.nn.Conv2d) else target
              for layer,target in zip(old.align_module,tc)]
        if sc != [64,128,256] or tc != [128,256,512]:
            raise ValueError(f'Unexpected student/teacher channels {sc}, {tc}')
        module = InstanceFeatureLoss(sc,tc,self.anchor_config).to(self.device)
        self.kd_feature_loss = module
        optimizer = super(core.AnchorDistiller,self).build_optimizer(model,feature_model=module,**kwargs)
        gate = list(module.gate.parameters()); ids = {id(p) for p in gate}
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if id(p) not in ids]
        optimizer.add_param_group({'params':gate,'lr':kwargs['lr']*self.anchor_config.gate_lr_scale,'weight_decay':0.})
        return optimizer

    def preprocess_batch(self,batch):
        batch = super().preprocess_batch(batch)
        self.kd_feature_loss.batch = batch
        return batch

    def _setup_train(self,world_size):
        super()._setup_train(world_size)
        student,teacher = self.model.model[-1],self.teacher_model.model[-1]
        for head in (student,teacher):
            if (int(head.nc),int(head.reg_max),tuple(int(v) for v in head.stride)) != (20,16,(8,16,32)):
                raise ValueError('Incompatible detection head')
        c=self.anchor_config
        self.kd_logical_loss=ResponseLoss(self,c.response_temperature,c.response_conf,c.response_iou,c.response_warmup).to(self.device)
        path=self.save_dir/'anchor_protocol.json'
        protocol=json.loads(path.read_text(encoding='utf-8'))
        protocol.update({'version':'single_moe150','single_stage':True,'stage1_checkpoint':None,
            'initialization':'seeded YOLOv8n yaml; pretrained=False; freshly initialized MGD',
            'response_ratio':c.response_ratio,'response_warmup':c.response_warmup,
            'gate_note':'feature gate query excludes the fixed response branch'})
        path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
