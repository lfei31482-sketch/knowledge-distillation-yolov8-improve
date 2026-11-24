# knowledge-distillation-yolov8-improve

**MGD（Masked Generative Distillation）改进算法包**，基于本项目的 Ultralytics YOLOv8 8.3.171 修改版。基线对照见 [knowledge-distillation-yolov8-baseline](https://github.com/lfei31482-sketch/knowledge-distillation-yolov8-baseline)。

## 结构

- `ultralytics/` — 修改后的 YOLOv8 蒸馏支持基础（蒸馏损失 `utils/distill_loss.py`、`DetectionDistiller` 等）
- `moe_kd_v3/` — MoE 蒸馏核心模块（`AnchorDistiller`、预检、checkpoint 校验等）
- `single_moe150/` — 单阶段蒸馏配置（`SingleConfig`、`SingleDistiller`）
- `instance_moe_v5/` — 实例级特征损失（`InstanceFeatureLoss`）
- `instance_moe_response_v1/` — 响应蒸馏损失（`ResponseLoss`）
- `mgd_response150/` — **MGD 改进核心**（`ResponseOnlyDistiller`）
- `train_moe_v3.py` — MoE 蒸馏训练入口（含数据配置）
- `train_mgd_response150.py` / `run_mgd_response150.sh` — MGD 150 epoch 训练入口与启动脚本
- `README_MGD_RESPONSE150.md` — MGD 实验详细说明

## 运行

```bash
pip install -e .
bash run_mgd_response150.sh
```

默认教师权重 `runs/voc_teacher_yolov8s_seed0/weights/best.pt`、数据根 `output/`（VOC），可用 `--teacher-weights` / `--data` 覆盖。

## 说明

- 不含数据集、教师/学生权重、训练日志与实验结果（见 `.gitignore`）。
