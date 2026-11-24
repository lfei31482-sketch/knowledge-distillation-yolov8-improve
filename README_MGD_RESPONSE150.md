# MGD + 响应蒸馏：150 轮 seed0 消融实验

目的：在单阶段、相同 150 轮预算下，检验上次 MoE 组合中响应蒸馏的贡献。固定完整三尺度 MGD 与响应蒸馏，关闭实例门控、CWD/Mimic 辅助专家、门控监督和辅助适配器预训练；其他设置沿用 single_moe150。训练从 seed0 的 YOLOv8n YAML 随机初始化开始，不加载上次训练模型。

门控和专家模块为了保持初始化及优化器结构一致仍在代码中构造，但不执行前向、不参与梯度更新；配置中的 mode=gated 是沿用的兼容字段，并不表示门控实际开启。响应蒸馏前 30 轮关闭，第 31–40 轮逐渐开启，权重 0.05。该实验是消融对照，不保证提升。

在已有 `moe_kd_v3`、`hierarchical_moe`、`instance_moe_v5`、`instance_moe_response_v1`、`train_moe_v3.py` 和 Ultralytics 环境的服务器项目根目录运行：

```bash
python -m mgd_response150.selftest cuda
bash run_mgd_response150.sh
```

输出 `runs/mgd_response150/mgd_response150_seed0/`；保存完整目录以便比较。中断后重复运行同一脚本可尝试从 `weights/moe_last.pt` 恢复，数据加载预取意味着续训不保证逐批完全复现未中断轨迹。若无完整训练状态而目标目录已存在，脚本会停止以避免覆盖。

仅使用验证集选择 `best.pt`。正式设置固定教师权重和数据 YAML 哈希、640、batch 8、FP32、SGD、150 轮；YAML 哈希并不证明图像和标注内容未变。`response_kd.csv` 从第 31 轮起每 50 batch 抽样，持续 selected_cells=0 表明响应分支可能未选中有效位置。该包不含数据集或教师权重。
