"""GT-box-filtered, teacher-quality-weighted YOLOv8 response distillation."""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
import moe_kd_v3.trainer as core


class ResponseLoss(nn.Module):
    """KL on class and DFL distributions at teacher-reliable GT-box cells."""
    def __init__(self, trainer, temperature=2.0, conf_min=0.25, iou_min=0.30, warmup=20):
        super().__init__()
        self.trainer = trainer
        self.temperature, self.conf_min, self.iou_min = temperature, conf_min, iou_min
        self.warmup = warmup
        self.nc, self.reg_max, self.strides = 20, 16, (8, 16, 32)

    def forward(self, student, teacher, batch):
        if not isinstance(student, (list, tuple)) or not isinstance(teacher, (list, tuple)):
            raise ValueError('Expected lists of three raw prediction scales')
        if len(student) != 3 or len(teacher) != 3:
            raise ValueError('Expected P3/P4/P5 prediction scales')
        imgs = batch['img']
        b, _, image_h, image_w = imgs.shape
        device = student[0].device
        ids = batch['batch_idx'].reshape(-1).to(device=device, dtype=torch.long)
        cls = batch['cls'].reshape(-1).to(device=device, dtype=torch.long)
        boxes = batch['bboxes'].reshape(-1, 4).to(device=device, dtype=torch.float32)
        ids_cpu = batch['batch_idx'].reshape(-1).cpu().tolist()
        cls_cpu = batch['cls'].reshape(-1).cpu().tolist()
        boxes_cpu = batch['bboxes'].reshape(-1, 4).cpu().tolist()
        if len(ids) != len(cls) or len(ids) != len(boxes):
            raise ValueError('GT field length mismatch')
        if len(ids) and ((ids < 0).any() or (ids >= b).any() or (cls < 0).any() or (cls >= self.nc).any()):
            raise ValueError('GT batch or class index is invalid')
        for scale, (sp,tp) in enumerate(zip(student,teacher)):
            if (sp.ndim != 4 or sp.shape != tp.shape or sp.shape[0] != b
                or sp.shape[1] != 4*self.reg_max+self.nc
                or sp.shape[-2]*self.strides[scale] != image_h
                or sp.shape[-1]*self.strides[scale] != image_w):
                raise ValueError(f'Incompatible raw prediction grid at scale {scale}')
        epoch = int(self.trainer.epoch)
        if epoch < self.warmup:
            return sum(x.sum()*0 for x in student)
        # Accumulate by GT across all scales; equalize valid instances within
        # each image and then give each image the same total weight.
        total_cls = student[0].new_zeros((len(ids),)) + student[0].sum() * 0
        total_dfl = student[0].new_zeros((len(ids),)) + student[0].sum() * 0
        total_weight = student[0].new_zeros((len(ids),))
        candidates = selected = 0
        scale_counts = []
        t = self.temperature
        for scale, (sp, tp) in enumerate(zip(student, teacher)):
            stride = self.strides[scale]
            if sp.shape != tp.shape or sp.ndim != 4 or sp.shape[0] != b or sp.shape[1] != 4 * self.reg_max + self.nc:
                raise ValueError(f'Raw prediction shape mismatch at scale {scale}: {sp.shape}, {tp.shape}')
            h, w = sp.shape[-2:]
            if h * stride != image_h or w * stride != image_w:
                raise ValueError('Raw predictions do not align with image size/strides')
            tp = tp.detach().float()
            sp = sp.float()
            yy, xx = torch.meshgrid(torch.arange(h, device=sp.device, dtype=sp.dtype) + .5,
                                    torch.arange(w, device=sp.device, dtype=sp.dtype) + .5, indexing='ij')
            with torch.no_grad():
                bins = torch.arange(self.reg_max, device=sp.device, dtype=sp.dtype).view(1, 1, -1, 1, 1)
                d = (tp[:, :4*self.reg_max].reshape(b, 4, self.reg_max, h, w).softmax(2) * bins).sum(2)
                tx1, ty1, tx2, ty2 = xx[None] - d[:,0], yy[None] - d[:,1], xx[None] + d[:,2], yy[None] + d[:,3]
                quality = sp.new_zeros((b, h, w))
                matched = torch.full((b, h, w), -1, device=sp.device, dtype=torch.long)
                candidate_mask = torch.zeros((b, h, w), device=sp.device, dtype=torch.bool)
                for j, (image_id, class_id, box) in enumerate(zip(ids_cpu, cls_cpu, boxes_cpu)):
                    k, c = int(image_id), int(class_id)
                    cx, cy, bw, bh = box
                    if not all(math.isfinite(v) for v in box) or bw <= 0 or bh <= 0:
                        continue
                    x1, y1 = (cx-bw/2)*w, (cy-bh/2)*h
                    x2, y2 = (cx+bw/2)*w, (cy+bh/2)*h
                    inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
                    candidate_mask[k] |= inside
                    inter = (tx2[k].clamp(max=x2)-tx1[k].clamp(min=x1)).clamp_min(0) * (ty2[k].clamp(max=y2)-ty1[k].clamp(min=y1)).clamp_min(0)
                    area_t = (tx2[k]-tx1[k]).clamp_min(0) * (ty2[k]-ty1[k]).clamp_min(0)
                    iou = inter / (area_t + max(x2-x1,0)*max(y2-y1,0) - inter + 1e-7)
                    confidence = tp[k,4*self.reg_max+c].sigmoid()
                    eligible = inside & (confidence >= self.conf_min) & (iou >= self.iou_min)
                    candidate_quality = (confidence * iou) * eligible
                    replace = candidate_quality > quality[k]
                    matched[k] = torch.where(replace, torch.full_like(matched[k], j), matched[k])
                    quality[k] = torch.maximum(quality[k], candidate_quality)
                scale_candidates = int(candidate_mask.sum().item())
                scale_selected = int((quality > 0).sum().item())
                candidates += scale_candidates
                selected += scale_selected
                scale_counts.append((scale_candidates, scale_selected))
            scls, tcls = sp[:,4*self.reg_max:]/t, tp[:,4*self.reg_max:]/t
            prob = tcls.sigmoid()
            kl_cls = (prob * (F.logsigmoid(tcls)-F.logsigmoid(scls)) + (1-prob)*(F.logsigmoid(-tcls)-F.logsigmoid(-scls))).mean(1) * (t*t)
            sdfl = sp[:,:4*self.reg_max].reshape(b,4,self.reg_max,h,w)/t
            tdfl = tp[:,:4*self.reg_max].reshape(b,4,self.reg_max,h,w)/t
            tlog = tdfl.log_softmax(2)
            kl_dfl = (tlog.exp() * (tlog-sdfl.log_softmax(2))).sum(2).mean(1) * (t*t)
            valid = matched.reshape(-1) >= 0
            if valid.any():
                index = matched.reshape(-1)[valid]
                total_cls = total_cls.index_add(0, index, (quality * kl_cls).reshape(-1)[valid])
                total_dfl = total_dfl.index_add(0, index, (quality * kl_dfl).reshape(-1)[valid])
                total_weight = total_weight.index_add(0, index, quality.reshape(-1)[valid])
        denom = total_weight.clamp_min(1e-12)
        instance_cls, instance_dfl = total_cls/denom, total_dfl/denom
        valid_gt = total_weight > 0
        per_image = torch.bincount(ids[valid_gt], minlength=b).clamp_min(1)
        cls_loss = (instance_cls[valid_gt] / per_image[ids[valid_gt]]).sum() / b
        dfl_loss = (instance_dfl[valid_gt] / per_image[ids[valid_gt]]).sum() / b
        ramp = min((epoch - self.warmup + 1) / 10.0, 1.0)
        loss = (cls_loss + dfl_loss) * b * ramp
        if self.trainer.batch_index % 50 == 0:
            core.append_csv(Path(self.trainer.save_dir)/'response_kd.csv', [{'epoch':epoch+1,
                'batch':self.trainer.batch_index, 'gt':len(ids),'candidate_cells':candidates,
                'selected_cells':selected, 'selection_ratio':selected/max(candidates,1),
                'quality_sum':float(total_weight.sum().detach()), 'valid_gt':int(valid_gt.sum().item()),
                'empty_selected':int(selected == 0),
                'p3_candidates':scale_counts[0][0], 'p3_selected':scale_counts[0][1],
                'p4_candidates':scale_counts[1][0], 'p4_selected':scale_counts[1][1],
                'p5_candidates':scale_counts[2][0], 'p5_selected':scale_counts[2][1],
                'cls_kl':float(cls_loss.detach()), 'dfl_kl':float(dfl_loss.detach()),
                'ramp':ramp,'unweighted_response_loss':float(loss.detach())}])
        return loss
