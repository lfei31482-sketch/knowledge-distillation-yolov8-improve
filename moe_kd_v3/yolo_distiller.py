# Ultralytics YOLO, AGPL-3.0 license.
# Vendored vanilla adapter from ultralytics/models/yolo/detect/distill.py.
# Source SHA256: ba689e064ed9478d1b2b96b22d7cf613f692ef9bf63d26cd784c4925619a564f
# Unused custom-kernel import removed; comments normalized with ast.unparse.
import csv
import sys, os, torch, math, time, warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pylab as plt
import torch.nn as nn
from torch import optim
from thop import clever_format
from functools import partial
from torch import distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from copy import copy, deepcopy
from pathlib import Path
import numpy as np
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, clean_url, colorstr, emojis, yaml_save, callbacks, __version__
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.checks import check_imgsz, print_args, check_amp
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, one_cycle, init_seeds, select_device
from ultralytics.utils.distill_loss import LogicalLoss, FeatureLoss

def get_temperature(*args, **kwargs):
    raise RuntimeError('MoE addon supports vanilla YOLOv8 only, not kernel-warehouse architectures')

def get_activation(feat, backbone_idx=-1):

    def hook(model, inputs, outputs):
        if backbone_idx != -1:
            for _ in range(5 - len(outputs)):
                outputs.insert(0, None)
            feat.append(outputs[backbone_idx])
        else:
            feat.append(outputs)
    return hook
GRADIENT_DIAGNOSTIC_EPOCHS = {5, 10, 20, 30, 50, 75, 100, 125, 150}
GRADIENT_DIAGNOSTIC_INTERVAL = 50
GRADIENT_DIAGNOSTIC_EPS = 1e-12

def should_record_feature_gradients(seed, epoch, batch_index):
    # The addon records selected-expert diagnostics through its own observer.
    return False

def _loss_gradients(loss, features):
    """Read float32 gradients without writing parameter or feature ``.grad`` fields."""
    gradients = torch.autograd.grad(loss.float(), features, retain_graph=True, allow_unused=True)
    return tuple((torch.zeros_like(feature, dtype=torch.float32) if gradient is None else gradient.detach().float() for feature, gradient in zip(features, gradients)))

def record_feature_gradient_diagnostics(csv_path, seed, feature_loss_type, epoch, batch_index, detection_loss, weighted_feature_loss, raw_student_features, weighted_scale_losses):
    """Record paper equations (12)-(15) on raw P3/P4/P5 student hook outputs."""
    if len(raw_student_features) != 3 or len(weighted_scale_losses) != 3:
        raise ValueError('Paper diagnostics require exactly three P3/P4/P5 feature scales')
    detection_gradients = _loss_gradients(detection_loss, raw_student_features)
    distillation_gradients = _loss_gradients(weighted_feature_loss, raw_student_features)
    det_sq = [gradient.square().sum() for gradient in detection_gradients]
    kd_sq = [gradient.square().sum() for gradient in distillation_gradients]
    dots = [(det_gradient * kd_gradient).sum() for det_gradient, kd_gradient in zip(detection_gradients, distillation_gradients)]
    det_norms = [value.sqrt() for value in det_sq]
    kd_norms = [value.sqrt() for value in kd_sq]
    total_det_norm = torch.stack(det_sq).sum().sqrt()
    total_kd_norm = torch.stack(kd_sq).sum().sqrt()
    global_cosine = torch.stack(dots).sum() / (total_det_norm * total_kd_norm).clamp_min(GRADIENT_DIAGNOSTIC_EPS)
    relative_norm = total_kd_norm / total_det_norm.clamp_min(GRADIENT_DIAGNOSTIC_EPS)
    loss_magnitudes = [loss.detach().float().abs() for loss in weighted_scale_losses]
    loss_total = torch.stack(loss_magnitudes).sum().clamp_min(GRADIENT_DIAGNOSTIC_EPS)
    kd_norm_total = torch.stack(kd_norms).sum().clamp_min(GRADIENT_DIAGNOSTIC_EPS)
    row = {'seed': int(seed), 'feature_loss_type': feature_loss_type, 'epoch': int(epoch), 'batch_index': int(batch_index), 'detection_loss': detection_loss.detach().float().item(), 'weighted_feature_loss': weighted_feature_loss.detach().float().item(), 'cgrad_global': global_cosine.item(), 'rnorm_global': relative_norm.item()}
    for label, scale_loss, qloss, det_norm, kd_norm, dot in zip(('p3', 'p4', 'p5'), weighted_scale_losses, (value / loss_total for value in loss_magnitudes), det_norms, kd_norms, dots):
        local_cosine = dot / (det_norm * kd_norm).clamp_min(GRADIENT_DIAGNOSTIC_EPS)
        row[f'loss_{label}'] = scale_loss.detach().float().item()
        row[f'qloss_{label}'] = qloss.item()
        row[f'det_grad_norm_{label}'] = det_norm.item()
        row[f'kd_grad_norm_{label}'] = kd_norm.item()
        row[f'cgrad_{label}'] = local_cosine.item()
        row[f'qgrad_{label}'] = (kd_norm / kd_norm_total).item()
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

class DetectionDistiller(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name
        self.wdir = self.save_dir / 'weights'
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))
        self.last, self.best = (self.wdir / 'last.pt', self.wdir / 'best.pt')
        self.save_period = self.args.save_period
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))
        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split('.')[-1] in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error 閴?{e}")) from e
        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None
        self.lf = None
        self.scheduler = None
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.logical_disloss = None
        self.feature_disloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING 閳跨媴绗?'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        self.model.nc = self.data['nc']
        self.model.names = self.data['names']
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = ('box_loss', 'cls_loss', 'dfl_loss')
        return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' * (4 + len(self.loss_names) + 2)) % ('Epoch', 'GPU_mem', *self.loss_names, 'log_loss', 'fea_loss', 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'], batch_idx=batch['batch_idx'], cls=batch['cls'].squeeze(-1), bboxes=batch['bboxes'], paths=batch['im_file'], fname=self.save_dir / f'train_batch{ni}.jpg', on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-05, iterations=100000.0, feature_model=None):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = ([], [], [])
        bn = tuple((v for k, v in nn.__dict__.items() if 'Norm' in k))
        if name == 'auto':
            LOGGER.info(f"{colorstr('optimizer:')} 'optimizer=auto' found, ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and determining best 'optimizer', 'lr0' and 'momentum' automatically... ")
            nc = getattr(model, 'nc', 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:
                    g[2].append(param)
                elif isinstance(module, bn):
                    g[1].append(param)
                else:
                    g[0].append(param)
        if feature_model is not None:
            for v in feature_model.modules():
                for p_name, p in v.named_parameters(recurse=0):
                    if p_name == 'bias':
                        g[2].append(p)
                    elif p_name == 'weight' and isinstance(v, bn):
                        g[1].append(p)
                    else:
                        g[0].append(p)
        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' not found in list of available optimizers [Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.")
        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)")
        return optimizer

    def setup_prune_model(self):
        ckpt = torch.load(self.model, map_location=self.device)
        model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        LOGGER.info(f"{colorstr('Loading Prune Student Model form {}'.format(self.model))}")
        self.model = model
        self.model.info()
        return ckpt

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):
            return
        model, weights = (self.model, None)
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = weights.yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=self.pretrain_weights, verbose=RANK == -1)
        return ckpt

    def setup_teacher_model(self):
        """Load a trained teacher checkpoint and keep it frozen in evaluation mode."""
        teacher_weights = Path(self.args.teacher_weights)
        if not teacher_weights.is_file():
            raise FileNotFoundError(f'Teacher checkpoint not found: {teacher_weights}')
        ckpt = torch.load(teacher_weights, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise TypeError(f'Teacher checkpoint must be a dictionary: {teacher_weights}')
        teacher = ckpt.get('ema') if ckpt.get('ema') is not None else ckpt.get('model')
        if teacher is None:
            raise KeyError(f"Teacher checkpoint has neither 'ema' nor 'model': {teacher_weights}")
        self.teacher_model = teacher.float().to(self.device).eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)
        self.teacher_model.info()
        return ckpt

    def _setup_train(self, world_size):
        if self.args.prune_model:
            ckpt = self.setup_prune_model()
        else:
            LOGGER.info(f"{colorstr('SetUp Student Model:')}")
            ckpt = self.setup_model()
        LOGGER.info(f"{colorstr('SetUp Teacher Model:')}")
        _ = self.setup_teacher_model()
        self.model.to(self.device)
        self.set_model_attributes()
        self.model.criterion = self.model.init_criterion()
        freeze_list = self.args.freeze if isinstance(self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any((x in k for x in freeze_layer_names)):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING 閳跨媴绗?setting 'requires_grad=True' for frozen layer '{k}'. See ultralytics.engine.trainer for customization of frozen layers.")
                v.requires_grad = True
        self.amp = torch.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in (-1, 0):
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup
        if RANK > -1 and world_size > 1:
            dist.broadcast(self.amp, src=0)
        self.amp = bool(self.amp)
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        if self.batch_size == -1 and RANK == -1:
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()
        self.kd_logical_loss, self.kd_feature_loss = (None, None)
        if self.args.kd_loss_type == 'logical' or self.args.kd_loss_type == 'all':
            self.kd_logical_loss = LogicalLoss(self.args, self.model, self.args.logical_loss_type)
        if self.args.kd_loss_type == 'feature' or self.args.kd_loss_type == 'all':
            s_feature, t_feature = ([], [])
            hooks = []
            self.teacher_kd_layers, self.student_kd_layers = (self.args.teacher_kd_layers.split(','), self.args.student_kd_layers.split(','))
            s_feature_idx, t_feature_idx = ([], [])
            assert len(self.teacher_kd_layers) == len(self.student_kd_layers), f'teacher{self.teacher_kd_layers} and student{self.student_kd_layers} layers not equal..'
            for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                if '-' in t_layer:
                    t_layer_first, t_layer_second = t_layer.split('-')
                    t_feature_idx.append(int(t_layer_second) / 10)
                    hooks.append(de_parallel(self.teacher_model).model[int(t_layer_first)].register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second))))
                else:
                    hooks.append(de_parallel(self.teacher_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))
                    t_feature_idx.append(int(t_layer))
                if '-' in s_layer:
                    s_layer_first, s_layer_second = s_layer.split('-')
                    s_feature_idx.append(int(s_layer_second) / 10)
                    hooks.append(de_parallel(self.model).model[int(s_layer_first)].register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second))))
                else:
                    hooks.append(de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))
                    s_feature_idx.append(int(s_layer))
            inputs = torch.randn((2, 3, self.args.imgsz, self.args.imgsz)).to(self.device)
            with torch.no_grad():
                _ = self.teacher_model(inputs)
                _ = self.model(inputs)
            s_feature_sort_idx, t_feature_sort_idx = (sorted(s_feature_idx), sorted(t_feature_idx))
            s_feature_idx = [s_feature_sort_idx.index(i) for i in s_feature_idx]
            t_feature_idx = [t_feature_sort_idx.index(i) for i in t_feature_idx]
            self.kd_feature_loss = FeatureLoss([s_feature[i].size(1) for i in s_feature_idx], [t_feature[i].size(1) for i in t_feature_idx], distiller=self.args.feature_loss_type).to(self.device)
            for hook in hooks:
                hook.remove()
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model, name=self.args.optimizer, lr=self.args.lr0, momentum=self.args.momentum, decay=weight_decay, iterations=iterations, feature_model=self.kd_feature_loss if isinstance(self.kd_feature_loss, FeatureLoss) else None)
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = (EarlyStopping(patience=self.args.patience), False)
        self.resume_training(None)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\nUsing {self.train_loader.num_workers * (world_size or 1)} dataloader workers\nLogging results to {colorstr('bold', self.save_dir)}\nStarting training for {self.epochs} epochs...")
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            self.teacher_model.eval()
            if self.args.kd_loss_type in ['feature', 'all']:
                self.kd_feature_loss.train()
                hooks = []
                s_feature, t_feature = ([], [])
                s_feature_idx, t_feature_idx = ([], [])
                for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                    if '-' in t_layer:
                        t_layer_first, t_layer_second = t_layer.split('-')
                        t_feature_idx.append(int(t_layer_second) / 10)
                        hooks.append(de_parallel(self.teacher_model).model[int(t_layer_first)].register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second))))
                    else:
                        hooks.append(de_parallel(self.teacher_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))
                        t_feature_idx.append(int(t_layer))
                    if '-' in s_layer:
                        s_layer_first, s_layer_second = s_layer.split('-')
                        s_feature_idx.append(int(s_layer_second) / 10)
                        hooks.append(de_parallel(self.model).model[int(s_layer_first)].register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second))))
                    else:
                        hooks.append(de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))
                        s_feature_idx.append(int(s_layer))
                s_feature_sort_idx, t_feature_sort_idx = (sorted(s_feature_idx), sorted(t_feature_idx))
                s_feature_idx = [s_feature_sort_idx.index(i) for i in s_feature_idx]
                t_feature_idx = [t_feature_sort_idx.index(i) for i in t_feature_idx]
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == self.epochs - self.args.close_mosaic:
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.logical_disloss = torch.zeros(1, device=self.device)
            self.feature_disloss = torch.zeros(1, device=self.device)
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                if hasattr(de_parallel(self.model), 'net_update_temperature'):
                    temp = get_temperature(i + 1, epoch, len(self.train_loader), temp_epoch=20, temp_init_value=1.0)
                    de_parallel(self.model).net_update_temperature(temp)
                if self.args.kd_loss_decay == 'constant':
                    distill_decay = 1.0
                elif self.args.kd_loss_decay == 'cosine':
                    eta_min, base_ratio, T_max = (0.01, 1.0, 10)
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
                elif self.args.kd_loss_decay == 'linear':
                    distill_decay = (1 - math.cos(i * math.pi / len(self.train_loader))) / 2 * (0.01 - 1) + 1
                elif self.args.kd_loss_decay == 'cosine_epoch':
                    eta_min, base_ratio, T_max = (0.01, 1.0, 10)
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * ni / T_max)) / 2
                elif self.args.kd_loss_decay == 'linear_epoch':
                    distill_decay = (1 - math.cos(ni * math.pi / (self.epochs * nb))) / 2 * (0.01 - 1) + 1
                record_gradients = self.kd_feature_loss is not None and should_record_feature_gradients(self.args.seed, epoch + 1, i)
                gradient_diagnostic_payload = None
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    pred = de_parallel(self.model).predict(batch['img'])
                    de_parallel(self.teacher_model).eval()
                    with torch.no_grad():
                        t_pred = de_parallel(self.teacher_model).predict(batch['img'])
                        if isinstance(t_pred, tuple):
                            t_pred = t_pred[1]
                    main_loss, self.loss_items = de_parallel(self.model).criterion(pred, batch)
                    main_loss = main_loss.sum()
                    log_distill_loss = torch.zeros(1, device=self.device)
                    fea_distill_loss = torch.zeros(1, device=self.device)
                    if self.kd_logical_loss is not None:
                        if type(pred) is dict and type(t_pred) is dict:
                            log_distill_loss = self.kd_logical_loss(pred['one2one'], t_pred['one2one'], batch, True) * self.args.logical_loss_ratio
                            log_distill_loss += self.kd_logical_loss(pred['one2many'], t_pred['one2many'], batch) * self.args.logical_loss_ratio * 0.5
                        else:
                            log_distill_loss = self.kd_logical_loss(pred, t_pred, batch) * self.args.logical_loss_ratio
                    if self.kd_feature_loss is not None:
                        raw_student_features = [s_feature[index] for index in s_feature_idx]
                        raw_teacher_features = [t_feature[index] for index in t_feature_idx]
                        fea_distill_loss = self.kd_feature_loss(raw_student_features, raw_teacher_features) * self.args.feature_loss_ratio
                        scale_losses = tuple(self.kd_feature_loss.last_scale_losses)
                        self.kd_feature_loss.last_scale_losses = ()
                        self.kd_feature_loss.feature_loss.last_scale_losses = ()
                        if record_gradients:
                            weighted_scale_losses = tuple((scale_loss * self.args.feature_loss_ratio * distill_decay for scale_loss in scale_losses))
                    self.loss = main_loss + (log_distill_loss + fea_distill_loss) * distill_decay
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    log_loss_for_logging = (log_distill_loss * distill_decay).detach()
                    self.logical_disloss = (self.logical_disloss * i + log_loss_for_logging) / (i + 1) if self.logical_disloss is not None else log_distill_loss
                    weighted_fea_loss = fea_distill_loss * distill_decay
                    if record_gradients:
                        gradient_diagnostic_payload = (main_loss, weighted_fea_loss, raw_student_features, weighted_scale_losses)
                    fea_loss_for_logging = (fea_distill_loss * distill_decay).detach()
                    self.feature_disloss = (self.feature_disloss * i + fea_loss_for_logging) / (i + 1)
                if gradient_diagnostic_payload is not None:
                    record_feature_gradient_diagnostics(csv_path=self.save_dir / 'gradient_diagnostics.csv', seed=self.args.seed, feature_loss_type=self.args.feature_loss_type, epoch=epoch + 1, batch_index=i, detection_loss=gradient_diagnostic_payload[0], weighted_feature_loss=gradient_diagnostic_payload[1], raw_student_features=gradient_diagnostic_payload[2], weighted_scale_losses=gradient_diagnostic_payload[3])
                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                mem = f'{(torch.cuda.memory_reserved() / 1000000000.0 if torch.cuda.is_available() else 0):.3g}G'
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                logical_dislosses = self.logical_disloss if loss_len > 1 else torch.unsqueeze(self.logical_disloss, 0)
                feature_dislosses = self.feature_disloss if loss_len > 1 else torch.unsqueeze(self.feature_disloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(('%11s' * 2 + '%11.4g' * (2 + loss_len + 2)) % (f'{epoch + 1}/{self.epochs}', mem, *losses, *logical_dislosses, *feature_dislosses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                self.run_callbacks('on_train_batch_end')
                if self.kd_feature_loss is not None:
                    s_feature.clear()
                    t_feature.clear()
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')
            if self.kd_feature_loss is not None:
                for hook in hooks:
                    hook.remove()
            if RANK in (-1, 0):
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = epoch + 1 == self.epochs or self.stopper.possible_stop
                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                fea_loss_value = self.feature_disloss.detach().item() if isinstance(self.feature_disloss, torch.Tensor) else float(self.feature_disloss)
                log_loss_value = self.logical_disloss.detach().item() if isinstance(self.logical_disloss, torch.Tensor) else float(self.logical_disloss)
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), 'train/log_loss_weighted': round(log_loss_value, 6), 'train/fea_loss_weighted': round(fea_loss_value, 6), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)
                if self.args.save or epoch + 1 == self.epochs:
                    self.save_model()
                    self.run_callbacks('on_model_save')
            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break
        if RANK in (-1, 0):
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def distill(self, weights=None):
        self.pretrain_weights = weights
        self.train()
