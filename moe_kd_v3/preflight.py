"""Read-only source/environment/data validation before a server queue."""
import argparse, json, os
from pathlib import Path
os.environ['VOC_OFFLINE_TRAINING']='1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path.cwd()/'.moe_yolo_config'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',default='runs/voc_teacher_yolov8s_seed0/weights/best.pt')
    p.add_argument('--data-root',default='output');p.add_argument('--device',default='0');a=p.parse_args()
    import torch
    import ultralytics
    from .trainer import AnchorDistiller,file_hash
    source=Path(ultralytics.__file__).resolve()
    if source.parent != Path.cwd()/'ultralytics':raise RuntimeError(f'Imported wrong ultralytics: {source}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    teacher=Path(a.teacher)
    if not teacher.is_file():raise FileNotFoundError(teacher)
    ckpt=torch.load(teacher,map_location='cpu',weights_only=False)
    model=ckpt.get('ema') if ckpt.get('ema') is not None else ckpt['model']
    if model.model[-1].nc!=20:raise ValueError('Teacher must be trained on VOC 20 classes')
    counts={};zero_size=[];data=Path(a.data_root)
    for split,expected in [('train',13240),('val',3311),('test',4952)]:
        images={p.stem for p in (data/'images'/split).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')}
        labels={p.stem for p in (data/'labels'/split).glob('*.txt')}
        if len(images)!=expected or not images<=labels:
            raise ValueError(f'{split}: expected {expected} paired images, got {len(images)} images, {len(images-labels)} missing labels')
        counts[split]=len(images)
        zero_size.extend(str(p) for p in (data/'images'/split).iterdir() if p.is_file() and p.stat().st_size==0)
    print(json.dumps({'source':str(source),'torch':torch.__version__,'cuda':torch.version.cuda,
                      'gpu':torch.cuda.get_device_name(int(a.device)),'teacher_sha256':file_hash(teacher),
                      'VOC_images':counts,'zero_size_images_preserved':zero_size,
                      'note':'Training skips corrupt images consistently; this preflight does not decode all JPEGs.',
                      'result':'PASS'},indent=2))


if __name__=='__main__':main()
