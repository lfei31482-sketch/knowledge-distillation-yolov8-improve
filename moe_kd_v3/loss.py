"""MGD is never routed, normalized or diluted. Routing controls one residual only."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F

AUX = ('cwd', 'tafd', 'at', 'mimic')
CHOICES = ('none',) + AUX


@dataclass
class AnchorConfig:
    mode: str = 'gated'  # mgd, fixed, random, gated
    expert: str = 'cwd'
    seed: int = 0
    hidden: int = 64
    warmup_epochs: int = 30
    ramp_epochs: int = 10
    aux_fraction: float = .1
    adapter_pretrain: float = .01
    probe_every: int = 200
    probe_samples: int = 2
    virtual_lr: float = .01
    min_gain: float = 1e-8
    min_alignment: float = .01
    scale_ema: float = .1
    max_scale: float = 1e4
    task_coef: float = .1
    gate_lr_scale: float = .1
    exploration: float = .05
    mgd_alpha: float = 2e-5
    mask_drop: float = .65

    def validate(self):
        if self.mode not in ('mgd', 'fixed', 'random', 'gated') or self.expert not in AUX:
            raise ValueError('Unknown anchor mode / residual expert')
        if min(self.hidden, self.probe_every, self.probe_samples) < 1:
            raise ValueError('Positive hidden/probe settings required')
        if min(self.warmup_epochs, self.ramp_epochs, self.adapter_pretrain, self.min_gain) < 0:
            raise ValueError('Negative schedule or margin')
        if not 0 <= self.aux_fraction <= .5 or not 0 < self.scale_ema <= 1:
            raise ValueError('Require auxiliary gradient budget in [0,.5], EMA in (0,1]')
        if not 0 <= self.exploration < 1 or not 0 <= self.mask_drop < 1:
            raise ValueError('Invalid exploration / mask')
        if min(self.virtual_lr, self.max_scale, self.gate_lr_scale, self.mgd_alpha) <= 0:
            raise ValueError('Positive virtual step / scale / learning rate required')
        if self.task_coef < 0:
            raise ValueError('Invalid task coefficient')
        if not 0 <= self.min_alignment < 1:raise ValueError('Invalid query alignment margin')


def auxiliary_vector(name, s, t):
    t = t.detach()
    if name == 'mimic':
        return (s-t).square().flatten(1).mean(1)
    if name == 'at':
        a = F.normalize(s.square().mean(1).flatten(1), dim=1)
        b = F.normalize(t.square().mean(1).flatten(1), dim=1)
        return (a-b).square().mean(1)
    if name == 'cwd':
        a, b = s.flatten(2).log_softmax(-1), t.flatten(2).log_softmax(-1)
        return (b.exp()*(b-a)).sum(-1).mean(1)
    if name == 'tafd':
        attention = t.abs().mean(1, keepdim=True)
        u = (attention/(attention.mean((2, 3), keepdim=True)+1e-6)).clamp(max=10).square()
        w = u/(u.mean((2, 3), keepdim=True)+1e-6)
        return (w*(s-t).square()).flatten(1).mean(1)
    raise ValueError(name)


class Aligner(nn.Module):
    def __init__(self, sc, tc):
        super().__init__()
        self.align_module = nn.ModuleList([nn.Conv2d(s,t,1) if s != t else nn.Identity()
                                           for s,t in zip(sc,tc)])
        self.norm = nn.ModuleList([nn.BatchNorm2d(t, affine=False) for t in tc])

    def forward(self, sf, tf):
        result = []
        for l,(s,t) in enumerate(zip(sf,tf)):
            s = self.align_module[l](s)
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode='bilinear', align_corners=False)
            result.append((self.norm[l](s), self.norm[l](t.detach())))
        return result


class RawMGD(nn.Module):
    """Same reductions/BN/masks as this repository's FeatureLoss(distiller='mgd')."""
    def __init__(self, sc, tc, alpha=2e-5, mask_drop=.65):
        super().__init__()
        self.aligner = Aligner(sc,tc)
        self.generators = nn.ModuleList([nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(),
                                                       nn.Conv2d(c,c,3,padding=1)) for c in tc])
        self.alpha, self.mask_drop = alpha, mask_drop

    def forward(self, sf, tf):
        result = []
        for l,(s,t) in enumerate(self.aligner(sf,tf)):
            # Preserve the original repository's CPU RNG mask convention as well.
            keep = (torch.rand((s.shape[0],1,*s.shape[-2:])).to(s.device) <= 1-self.mask_drop).to(s.dtype)
            v = (self.generators[l](s*keep)-t).square().flatten(1).sum(1)*self.alpha
            result.append(v)
        return result


class Gate(nn.Module):
    def __init__(self, sc, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*sum(sc),hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden,len(CHOICES)))
        with torch.no_grad():
            self.net[-1].weight.mul_(.1)
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = 1.5  # conservative prior; none is a valid decision

    def forward(self, features):
        stats = []
        for f in features:
            x = f.detach().float()
            stats += [x.mean((2,3)), x.square().mean((2,3)).add(1e-8).sqrt()]
        return self.net(torch.cat(stats,1)).softmax(-1)


class AnchorFeatureLoss(nn.Module):
    def __init__(self, sc, tc, config=None):
        super().__init__()
        self.config = config or AnchorConfig()
        self.config.validate()
        self.mgd = RawMGD(sc,tc,self.config.mgd_alpha,self.config.mask_drop)
        # Auxiliary construction cannot change the main student's MGD mask RNG.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed+7907)
            self.adapters = nn.ModuleDict({name:Aligner(sc,tc) for name in AUX})
            self.gate = Gate(sc,self.config.hidden)
        self.register_buffer('aux_scales', torch.ones(3,len(AUX)))
        self.register_buffer('scale_updates', torch.zeros((),dtype=torch.long))
        self.epoch = 0
        self.last_scale_losses, self.trace, self.last_stats = (), {}, {}
        self.task_supervisor = None
        self.observer = None
        self._cache = {}
        self._router = None
        self._pending_router_state = None

    @property
    def feature_loss(self):
        return self

    def metadata(self):
        return {'config':asdict(self.config),'choice_order':CHOICES,
                'mgd_coefficient':1.0,'mgd_normalization':'none',
                'auxiliary_scales':self.aux_scales.detach().cpu().tolist()}

    def aux_vectors(self, name, sf, tf):
        if name not in self._cache:
            self._cache[name] = [auxiliary_vector(name,s,t) for s,t in self.adapters[name](sf,tf)]
        return self._cache[name]

    def fraction(self):
        c = self.config
        if c.mode == 'mgd' or self.epoch < c.warmup_epochs:
            return 0.
        ramp = min((self.epoch-c.warmup_epochs+1)/max(c.ramp_epochs,1),1.)
        return c.aux_fraction*ramp

    def router_state(self):
        return self._router.get_state() if self._router else self._pending_router_state

    def set_router_state(self, value):
        self._pending_router_state = value.cpu() if value is not None else None
        self._router = None

    def _sample(self, probabilities):
        if self._router is None:
            self._router = torch.Generator(device=probabilities.device)
            self._router.manual_seed(self.config.seed+104729)
            if self._pending_router_state is not None:
                self._router.set_state(self._pending_router_state)
                self._pending_router_state = None
        return torch.multinomial(probabilities.detach(),1,generator=self._router).squeeze(1)

    def forward(self, student_features, teacher_features, batch=None):
        c = self.config
        with torch.autocast(device_type=student_features[0].device.type,enabled=False):
            sf = [x.float() for x in student_features]
            tf = [x.detach().float() for x in teacher_features]
            self._cache = {}
            base_vectors = self.mgd(sf,tf)
            base = [v.mean() for v in base_vectors]
            if not all(torch.isfinite(v).all() for v in base_vectors):
                raise FloatingPointError('Non-finite raw MGD')
            probs = self.gate(sf)
            task = base[0].new_zeros(())
            shadow = base[0].new_zeros(())
            fraction = self.fraction()
            active = fraction > 0
            if active and self.task_supervisor is not None:
                task = self.task_supervisor(sf,tf,base_vectors,probs)
            if not active and c.mode != 'mgd' and c.aux_fraction > 0 and c.adapter_pretrain > 0:
                # Warm up independent adapters, never the student, with auxiliary losses.
                # These four experts are only all evaluated in warmup / periodic probes.
                detached = [s.detach() for s in sf]
                for name in AUX:
                    for s,t in self.adapters[name](detached,tf):
                        v = auxiliary_vector(name,s,t).mean()
                        shadow = shadow + v/v.detach().clamp_min(1e-12)
                shadow = shadow*c.adapter_pretrain
            ids = torch.zeros(sf[0].shape[0],device=sf[0].device,dtype=torch.long)
            if active:
                if c.mode == 'fixed':
                    ids.fill_(CHOICES.index(c.expert))
                elif c.mode == 'random':
                    ids = self._sample(torch.ones_like(probs)/len(CHOICES))
                elif c.mode == 'gated':
                    ids = self._sample((1-c.exploration)*probs+c.exploration/len(CHOICES))
            residual = [b.new_zeros(()) for b in base]
            records = []
            if active:
                if self.scale_updates.item() == 0:
                    raise RuntimeError('Auxiliary scales were not calibrated before routing')
                for e,name in enumerate(AUX):
                    index = (ids==e+1).nonzero(as_tuple=True)[0]
                    if not index.numel():continue
                    vv = self.aux_vectors(name,sf,tf)
                    for l,v in enumerate(vv):
                        coefficient = fraction*self.aux_scales[l,e].detach()
                        contribution = v[index]*coefficient
                        residual[l] = residual[l]+contribution.sum()/ids.numel()
                        records.append((l,e,index,v[index],contribution))
            extra = (c.task_coef*task+shadow)/3
            self.last_scale_losses = tuple(b+a+extra for b,a in zip(base,residual))
            result = sum(self.last_scale_losses)
            if not torch.isfinite(result):raise FloatingPointError('Non-finite anchor objective')
            self.last_stats = {'mgd_raw':sum(base).detach().item(),
                               'aux_weighted':sum(residual).detach().item(),
                               'task_ce':task.detach().item(),'adapter_pretrain':shadow.detach().item(),
                               'aux_fraction':fraction,'none_rate':(ids==0).float().mean().item(),
                               'gate_entropy':-(probs*probs.clamp_min(1e-12).log()).sum(1).mean().detach().item()}
            self.trace = {'ids':ids.detach(),'probs':probs.detach(),'records':records,
                          'base_vectors':base_vectors,'base_loss':sum(base),
                          'features':sf,'fraction':fraction,'active':active}
            if self.observer is not None:self.observer(self.trace)
            self._cache = {}
            return result
