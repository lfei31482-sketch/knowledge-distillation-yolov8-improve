"""Read-only, one-step plain-SGD surrogate; never updates the live optimizer."""
from contextlib import contextmanager
from collections import OrderedDict
import torch
from torch.func import functional_call


def clipped_gradients(gradients, maximum=10.):
    present = [g for g in gradients.values() if g is not None]
    if not present:return gradients
    norm = torch.stack([g.detach().float().square().sum() for g in present]).sum().sqrt()
    factor = (maximum/(norm+1e-6)).clamp(max=1.)
    return {k:g.detach()*factor if g is not None else None for k,g in gradients.items()}


@contextmanager
def isolated_eval(model):
    flags = [(m,m.training) for m in model.modules()]
    hooks = [(m,m._forward_hooks) for m in model.modules() if m._forward_hooks]
    # Detect caches are ordinary attributes, outside named_buffers.
    caches = [(m,k,getattr(m,k)) for m in model.modules() for k in ('shape','anchors','strides')
              if hasattr(m,k)]
    try:
        for m,_ in flags:m.training = False
        for m,_ in hooks:m._forward_hooks = OrderedDict()
        yield
    finally:
        for m,k,v in caches:setattr(m,k,v)
        for m,h in hooks:m._forward_hooks = h
        for m,flag in flags:m.training = flag


def virtual_loss(model, criterion, query, base_gradients, auxiliary_gradients=None, step=.01):
    """Scalar query detection loss after a virtual update of student parameters.

    This proxy has no momentum/accumulation and is not an exact replay of the live
    SGD optimizer. Both candidates use the same clipping, step, BN buffers and data.
    Query inputs come from a previous TRAINING batch, never validation or test.
    """
    combined = {}
    for name,_ in model.named_parameters():
        b = base_gradients.get(name)
        a = auxiliary_gradients.get(name) if auxiliary_gradients else None
        combined[name] = a if b is None else b if a is None else b+a
    combined = clipped_gradients(combined)
    parameters = {name:p.detach() if combined[name] is None else p.detach()-step*combined[name]
                  for name,p in model.named_parameters()}
    buffers = {name:b.detach().clone() for name,b in model.named_buffers()}
    devices = [next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    with isolated_eval(model), torch.random.fork_rng(devices=devices), torch.no_grad():
        pred = functional_call(model,(parameters,buffers),(query['img'],),strict=False)
        loss = criterion(pred,query)[0].sum()/query['img'].shape[0]
    if not torch.isfinite(loss):raise FloatingPointError('Virtual query detection loss is not finite')
    return float(loss)


def choice_target(gains, minimum_gain):
    """None is selected unless an auxiliary improves the base query loss by margin."""
    best = max(range(len(gains)),key=gains.__getitem__)
    return best+1 if gains[best] > minimum_gain else 0


def virtual_loss_and_gradient(model,criterion,query,base_gradients,step=.01):
    """Query gradient at virtual BASE weights, without touching live parameter grads."""
    clipped=clipped_gradients(base_gradients)
    parameters={name:(p.detach()-step*clipped[name] if clipped.get(name) is not None else p.detach()).requires_grad_(p.requires_grad)
                for name,p in model.named_parameters()}
    buffers={name:b.detach().clone() for name,b in model.named_buffers()}
    trainable=[(name,p) for name,p in parameters.items() if p.requires_grad]
    devices=[next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    with isolated_eval(model),torch.random.fork_rng(devices=devices),torch.enable_grad():
        pred=functional_call(model,(parameters,buffers),(query['img'],),strict=False)
        loss=criterion(pred,query)[0].sum()/query['img'].shape[0]
        grads=torch.autograd.grad(loss,[p for _,p in trainable],allow_unused=True)
    if not torch.isfinite(loss):raise FloatingPointError('Non-finite virtual base query')
    return float(loss.detach()),{name:g.detach() if g is not None else None for (name,_),g in zip(trainable,grads)}


def marginal_gain(base_gradients,auxiliary_gradients,query_gradients,base_query_loss,step=.01):
    """Stable first-order gain around the virtual base; scalar sums use FP64.

    Uses the difference of CLIPPED candidate/base steps. This is still a local
    approximation, not an observed detection AP gain or an exact optimizer replay.
    """
    combined={name:(a if b is None else b if a is None else b+a)
              for name in dict.fromkeys([*base_gradients,*auxiliary_gradients])
              for b,a in [(base_gradients.get(name),auxiliary_gradients.get(name))]}
    bclip=clipped_gradients(base_gradients);cclip=clipped_gradients(combined)
    dot=qnorm=dnorm=None
    for name,q in query_gradients.items():
        if q is None:continue
        b,c=bclip.get(name),cclip.get(name)
        if b is None and c is None:continue
        delta=c if b is None else -b if c is None else c-b
        q=q.double();delta=delta.double()
        terms=((q*delta).sum(),q.square().sum(),delta.square().sum())
        if dot is None:dot,qnorm,dnorm=terms
        else:dot=dot+terms[0];qnorm=qnorm+terms[1];dnorm=dnorm+terms[2]
    if dot is None:return 0.,0.
    gain=float(step*dot)/max(abs(base_query_loss),1e-8)
    alignment=float(dot/(qnorm.sqrt()*dnorm.sqrt()).clamp_min(1e-30))
    if not all(__import__('math').isfinite(x) for x in (gain,alignment)):
        raise FloatingPointError('Non-finite marginal query gain')
    return gain,alignment
