"""Training-only raw-MGD anchor and optional residual expert, single-GPU FP32."""
import argparse,json,os,hashlib
from pathlib import Path
os.environ['VOC_OFFLINE_TRAINING']='1'
os.environ['WANDB_MODE']='disabled'
os.environ['COMET_MODE']='DISABLED'
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parent/'.moe_yolo_config'))


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['mgd','fixed','random','gated'],default='gated')
    p.add_argument('--expert',choices=['cwd','tafd','at','mimic'],default='cwd')
    p.add_argument('--teacher-weights',default='runs/voc_teacher_yolov8s_seed0/weights/best.pt')
    p.add_argument('--data');p.add_argument('--data-root',default='output')
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--imgsz',type=int,default=640);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',default='0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--project',default='runs/moe_voc_v3');p.add_argument('--name')
    p.add_argument('--lambda-mgd',type=float,default=1.)
    p.add_argument('--warmup-epochs',type=int,default=30);p.add_argument('--ramp-epochs',type=int,default=10)
    p.add_argument('--aux-fraction',type=float,default=.1);p.add_argument('--adapter-pretrain',type=float,default=.01)
    p.add_argument('--probe-every',type=int,default=200);p.add_argument('--probe-samples',type=int,default=2)
    p.add_argument('--virtual-lr',type=float,default=.01);p.add_argument('--min-gain',type=float,default=1e-8)
    p.add_argument('--min-alignment',type=float,default=.01)
    p.add_argument('--task-coef',type=float,default=.1);p.add_argument('--gate-lr-scale',type=float,default=.1)
    p.add_argument('--exploration',type=float,default=.05)
    p.add_argument('--route-every',type=int,default=1);p.add_argument('--diagnostic-every',type=int,default=200)
    p.add_argument('--resume-state');p.add_argument('--stop-after-epoch',type=int,default=0)
    p.add_argument('--fraction',type=float,default=1.,help='Engineering smoke runs only; formal=1')
    p.add_argument('--eval-only',action='store_true');p.add_argument('--weights')
    p.add_argument('--split',choices=['val','test'],default='val')
    return p.parse_args()


def data_yaml(a):
    if a.data:
        p=Path(a.data).resolve()
        if not p.is_file():raise FileNotFoundError(p)
        return str(p)
    import yaml
    root=Path(a.data_root).resolve()
    for split in ('train','val','test'):
        for kind in ('images','labels'):
            if not (root/kind/split).is_dir():raise FileNotFoundError(root/kind/split)
    names=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
           'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    out=Path(a.project).resolve()/'datasets';out.mkdir(parents=True,exist_ok=True)
    p=out/('voc_'+hashlib.sha256(str(root).encode()).hexdigest()[:10]+'.yaml')
    content=yaml.safe_dump({'path':str(root),'train':'images/train','val':'images/val','test':'images/test',
                            'channels':3,'nc':20,'names':names},sort_keys=False)
    if p.exists() and p.read_text(encoding='utf-8')!=content:raise ValueError('Dataset YAML already exists with different content')
    if not p.exists():p.write_text(content,encoding='utf-8')
    return str(p)


def main():
    a=arguments()
    if ',' in a.device or int(os.environ.get('WORLD_SIZE','1'))>1:raise ValueError('Single process/GPU only')
    if a.lambda_mgd<=0 or min(a.route_every,a.diagnostic_every,a.workers)<0:raise ValueError('Invalid coefficient/logging')
    import torch
    torch.set_num_threads(min(torch.get_num_threads(),4))
    from ultralytics import YOLO
    from moe_kd_v3.loss import AnchorConfig
    from moe_kd_v3.trainer import AnchorDistiller
    data=data_yaml(a);name=a.name or f'anchor_{a.mode}_seed{a.seed}'
    if a.eval_only:
        if not a.weights:raise ValueError('--weights required')
        model=YOLO(a.weights)
        if any('gate' in k or 'generators' in k for k in model.model.state_dict()):raise ValueError('Use student-only best.pt')
        metrics=model.val(data=data,split=a.split,imgsz=a.imgsz,batch=a.batch,device=a.device,workers=a.workers,
                          half=False,plots=False,project=a.project,name=name+'_'+a.split)
        (Path(metrics.save_dir)/'metrics.json').write_text(json.dumps({'split':a.split,
             'weights':str(Path(a.weights).resolve()),'metrics':metrics.results_dict,
             'per_class_map':metrics.box.maps.tolist()},indent=2),encoding='utf-8');return
    if a.resume_state:
        s=torch.load(a.resume_state,map_location='cpu',weights_only=False)
        if s.get('format')!=3:raise ValueError('V1/V2 checkpoints cannot resume V3')
        if name!=s['args']['name'] or Path(a.project).resolve()!=Path(s['args']['project']).resolve():
            raise ValueError('Resume must use original project/name')
    elif (Path(a.project)/name).exists():raise FileExistsError('Choose a new name or resume: '+str(Path(a.project)/name))
    teacher=Path(a.teacher_weights).resolve()
    if not teacher.is_file():raise FileNotFoundError(teacher)
    cfg=AnchorConfig(mode=a.mode,expert=a.expert,seed=a.seed,warmup_epochs=a.warmup_epochs,
         ramp_epochs=a.ramp_epochs,aux_fraction=a.aux_fraction,adapter_pretrain=a.adapter_pretrain,
         probe_every=a.probe_every,probe_samples=a.probe_samples,virtual_lr=a.virtual_lr,min_gain=a.min_gain,
         min_alignment=a.min_alignment,
         task_coef=a.task_coef,gate_lr_scale=a.gate_lr_scale,exploration=a.exploration)
    cfg.validate()
    student=str(Path(__file__).resolve().parent/'ultralytics/cfg/models/v8/yolov8n.yaml')
    overrides=dict(model=student,data=data,imgsz=a.imgsz,epochs=a.epochs,batch=a.batch,workers=a.workers,
         device=a.device,seed=a.seed,project=str(Path(a.project).resolve()),name=name,exist_ok=bool(a.resume_state),
         pretrained=False,amp=False,optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,
         warmup_epochs=3.,warmup_bias_lr=0.,deterministic=True,patience=0,cache=False,plots=False,save=True,
         close_mosaic=10,fraction=a.fraction,mosaic=1.,mixup=0.,copy_paste=0.,hsv_h=.015,hsv_s=.7,
         hsv_v=.4,degrees=0.,translate=.1,scale=.5,shear=0.,perspective=0.,flipud=0.,fliplr=.5,
         teacher_weights=str(teacher),prune_model=False,kd_loss_type='feature',kd_loss_decay='constant',
         teacher_kd_layers='15,18,21',student_kd_layers='15,18,21',feature_loss_type='mimic',
         feature_loss_ratio=a.lambda_mgd)
    print('V3 RAW MGD ANCHOR: coefficient=1, no loss calibration; FP32; auxiliaries training-only',flush=True)
    print('MODE:',a.mode,'warmup:',a.warmup_epochs,'auxiliary gradient fraction:',a.aux_fraction,flush=True)
    trainer=AnchorDistiller(config=cfg,resume_state=a.resume_state,route_every=a.route_every,
                             diagnostic_every=a.diagnostic_every,overrides=overrides)
    def stop_at_boundary(t):
        pause=(Path(a.project)/'PAUSE_QUEUE').exists()
        if pause or (a.stop_after_epoch and t.epoch+1>=a.stop_after_epoch):
            t.stop=True;print(f'SAFE EPOCH STOP {t.epoch+1}; resume weights/moe_last.pt',flush=True)
    trainer.callbacks['on_fit_epoch_end'].append(stop_at_boundary)
    trainer.distill()


if __name__=='__main__':main()
