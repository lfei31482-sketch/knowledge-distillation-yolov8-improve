"""Verify inference checkpoint is an ordinary YOLOv8n (no auxiliary modules)."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path.cwd()/'.moe_yolo_config'))
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def verify(path, device='cpu'):
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    model = checkpoint.get('ema') if checkpoint.get('ema') is not None else checkpoint['model']
    assert type(model) is DetectionModel, type(model)
    assert not any('moe_kd' in type(m).__module__ for m in model.modules())
    assert not any(any(k in name.lower() for k in ('gate','generator','align_module'))
                   for name in model.state_dict())
    ref = DetectionModel('ultralytics/cfg/models/v8/yolov8n.yaml',nc=20,verbose=False)
    assert list(model.state_dict()) == list(ref.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in ref.parameters())
    # Reload through native YOLO API and exercise prediction. This requires no MoE trainer.
    loaded = YOLO(str(path)).model.float().to(device).eval()
    with torch.no_grad():
        output = loaded(torch.zeros(1,3,64,64,device=device))
    assert torch.isfinite(output[0]).all()
    print('PASS: native DetectionModel, exact YOLOv8n parameter topology, no gate/expert/adapter, forward finite')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(2);verify(args.checkpoint,args.device)
