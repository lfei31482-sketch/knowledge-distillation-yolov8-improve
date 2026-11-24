"""One 150-epoch VOC run from the MGD baseline's seeded random initialization."""
import argparse
import os
from pathlib import Path
os.environ['VOC_OFFLINE_TRAINING']='1'
os.environ['WANDB_MODE']='disabled'
os.environ['COMET_MODE']='DISABLED'
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parent/'.moe_yolo_config'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-weights',default='runs/voc_teacher_yolov8s_seed0/weights/best.pt')
    p.add_argument('--data-root',default='output'); p.add_argument('--data')
    p.add_argument('--project',default='runs/mgd_response150'); p.add_argument('--name',default='mgd_response150_seed0')
    p.add_argument('--device',default='0'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume-state'); p.add_argument('--stop-after-epoch',type=int,default=0)
    p.add_argument('--engineering-epochs',type=int,default=150); p.add_argument('--engineering-imgsz',type=int,default=640)
    p.add_argument('--engineering-batch',type=int,default=8); p.add_argument('--engineering-warmup',type=int,default=30)
    p.add_argument('--engineering-fraction',type=float,default=1.)
    a=p.parse_args()
    if ',' in a.device or int(os.environ.get('WORLD_SIZE','1'))>1: raise ValueError('Single GPU only')
    import torch
    torch.set_num_threads(min(torch.get_num_threads(),4))
    from train_moe_v3 import data_yaml
    from moe_kd_v3.trainer import file_hash
    from single_moe150.trainer import SingleConfig
    from mgd_response150.trainer import ResponseOnlyDistiller
    data=data_yaml(a); teacher=Path(a.teacher_weights).resolve()
    if not teacher.is_file(): raise FileNotFoundError(teacher)
    formal=(a.engineering_epochs,a.engineering_imgsz,a.engineering_batch,a.engineering_warmup,a.engineering_fraction)==(150,640,8,30,1.)
    if formal:
        if file_hash(teacher)!='adea16426e575044d9e184a5e51c9f5d123ebb1aa758de5dc99aeef5cda44e80':
            raise ValueError('Teacher does not match the 150-epoch MGD anchor')
        if file_hash(data)!='0eac5dcbb69cdddfd179c257406597cc43d5e92a3f1e8a163c866532acf41fe4':
            raise ValueError('Data YAML does not match the 150-epoch MGD anchor')
    else:
        print('ENGINEERING MODE: not a formal 150-epoch result',flush=True)
    run=Path(a.project).resolve()/a.name
    if not a.resume_state and run.exists(): raise FileExistsError(f'Run exists: {run}')
    if a.resume_state:
        state=torch.load(a.resume_state,map_location='cpu',weights_only=False)
        if not state.get('config',{}).get('single_stage') or state.get('format')!=3: raise ValueError('Not a single-stage checkpoint')
        if state['args']['name']!=a.name or Path(state['args']['project']).resolve()!=Path(a.project).resolve(): raise ValueError('Resume path mismatch')
    cfg=SingleConfig(warmup_epochs=a.engineering_warmup,ramp_epochs=10,aux_fraction=0.,adapter_pretrain=0.,task_coef=0.,
        probe_every=50,probe_samples=2,learning_rate=.01,response_warmup=a.engineering_warmup)
    cfg.validate()
    student=str(Path(__file__).resolve().parent/'ultralytics/cfg/models/v8/yolov8n.yaml')
    overrides=dict(model=student,data=data,imgsz=a.engineering_imgsz,epochs=a.engineering_epochs,
        batch=a.engineering_batch,workers=a.workers,device=a.device,seed=0,
        project=str(Path(a.project).resolve()),name=a.name,exist_ok=bool(a.resume_state),
        pretrained=False,amp=False,optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,
        warmup_epochs=3.,warmup_bias_lr=0.,deterministic=True,patience=0,cache=False,plots=False,
        save=True,close_mosaic=10,fraction=a.engineering_fraction,mosaic=1.,mixup=0.,copy_paste=0.,
        hsv_h=.015,hsv_s=.7,hsv_v=.4,degrees=0.,translate=.1,scale=.5,shear=0.,perspective=0.,
        flipud=0.,fliplr=.5,teacher_weights=str(teacher),prune_model=False,kd_loss_type='all',
        logical_loss_type='l2',logical_loss_ratio=cfg.response_ratio,kd_loss_decay='constant',
        teacher_kd_layers='15,18,21',student_kd_layers='15,18,21',feature_loss_type='mimic',feature_loss_ratio=1.)
    trainer=ResponseOnlyDistiller(config=cfg,resume_state=a.resume_state,route_every=0,diagnostic_every=0,overrides=overrides)
    def stop(t):
        if (Path(a.project)/'PAUSE_QUEUE').exists() or (a.stop_after_epoch and t.epoch+1>=a.stop_after_epoch): t.stop=True
    trainer.callbacks['on_fit_epoch_end'].append(stop)
    print(f'MGD + RESPONSE ONLY, NO GATE/EXPERTS: {a.engineering_epochs} epochs, seed0, no stage-one load',flush=True)
    trainer.distill()

if __name__=='__main__': main()
