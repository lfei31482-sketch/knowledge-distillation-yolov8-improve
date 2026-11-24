"""Isolated stage-two instance MoE with an extra response KD branch."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import moe_kd_v3.trainer as core
from moe_kd_v3.loss import AnchorConfig
from instance_moe_v5.loss import InstanceConfig
from instance_moe_v5.trainer import InstanceDistiller, addon_hash as v5_hash
from .response import ResponseLoss

VERSION = 'instance_moe_response_v1'
EXPECTED_V5_HASH = '239c13ab1a12363771d3005500110dfc04ddd3a68c2dee05f920a56eb92f9db9'
EXPECTED = {'teacher':'adea16426e575044d9e184a5e51c9f5d123ebb1aa758de5dc99aeef5cda44e80',
            'stage1':'5f87e032758e670540857a3f304456a88db990d77454072c07efb7833e3c559b',
            'aux':'332f139accf67bb909185fa1bd32e46d44cf424c82de9488af4ad3a7865bead7',
            'data':'0eac5dcbb69cdddfd179c257406597cc43d5e92a3f1e8a163c866532acf41fe4'}

@dataclass
class ResponseConfig(InstanceConfig):
    method_version: str = VERSION
    response_ratio: float = .05
    response_conf: float = .25
    response_iou: float = .30
    response_temperature: float = 2.
    response_warmup: int = 20

    def validate(self):
        AnchorConfig.validate(self)
        if self.method_version != VERSION or self.mode != 'gated':
            raise ValueError('Invalid response experiment version or mode')
        if self.response_ratio < 0 or self.response_warmup < 0 or not 0 <= self.response_conf <= 1 or not 0 <= self.response_iou <= 1:
            raise ValueError('Invalid response loss parameters')
        if min(self.gate_temperature, self.gate_target_temperature, self.learning_rate, self.response_temperature) <= 0:
            raise ValueError('Temperatures and learning rate must be positive')
        if bool(self.stage1_path) != bool(self.stage1_sha256):
            raise ValueError('Stage-one path/hash mismatch')
        for value in (self.stage1_sha256,self.stage1_aux_sha256):
            if value and len(value) != 64: raise ValueError('Invalid hash')

def addon_hash():
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256(v5_hash().encode())
    for name in ('instance_moe_response_v1/response.py','instance_moe_response_v1/trainer.py',
                 'train_instance_moe_response_v1.py', 'ultralytics/utils/loss.py',
                 'ultralytics/models/yolo/detect/distill.py'):
        h.update(name.encode()); h.update((root/name).read_bytes())
    return h.hexdigest()

def validate_formal(stage1, teacher, data):
    if v5_hash() != EXPECTED_V5_HASH:
        raise ValueError('Original v5 source differs from recorded probe50 experiment')
    for key,path in {'teacher':teacher,'stage1':stage1,'aux':stage1.parent/'moe_best.pt','data':data}.items():
        if core.file_hash(path) != EXPECTED[key]: raise ValueError(f'{key} hash differs from probe50')

class ResponseDistiller(InstanceDistiller):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        core.addon_hash = addon_hash

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        sh, th = self.model.model[-1], self.teacher_model.model[-1]
        if (int(sh.nc), int(sh.reg_max), tuple(int(v) for v in sh.stride)) != (20,16,(8,16,32)):
            raise ValueError('Student head is incompatible')
        if (int(th.nc), int(th.reg_max), tuple(int(v) for v in th.stride)) != (20,16,(8,16,32)):
            raise ValueError('Teacher head is incompatible')
        c = self.anchor_config
        self.kd_logical_loss = ResponseLoss(self, c.response_temperature, c.response_conf, c.response_iou,c.response_warmup).to(self.device)
        path = self.save_dir/'anchor_protocol.json'
        protocol = json.loads(path.read_text(encoding='utf-8'))
        protocol.update({'version':VERSION, 'response':{'ratio':c.response_ratio,'temperature':c.response_temperature,
            'conf_min':c.response_conf,'iou_min':c.response_iou,
            'selection':'GT box center cells; same-GT teacher class confidence and decoded DFL box IoU; overlaps select largest confidence*IoU',
            'weighting':'quality-weighted cells per GT; equal GT within image; equal images',
            'ramp':f'zero for {c.response_warmup} epochs; then linear rise over 10 epochs',
            'gate':'response branch fixed-weighted outside instance gate; gate task proxy excludes response loss',
            'inference':'no added model parameters'}})
        path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
