"""Test that the ablation never executes gate/experts, plus response checks."""
import sys
import torch
from single_moe150.trainer import SingleConfig
from single_moe150.selftest import main as upstream_test
from instance_moe_v5.loss import InstanceFeatureLoss

def main():
    device = sys.argv[1] if len(sys.argv)>1 else "cpu"
    cfg = SingleConfig(warmup_epochs=30, aux_fraction=0., adapter_pretrain=0., task_coef=0., learning_rate=.01)
    cfg.validate()
    for epoch in (0, 30, 149):
        torch.manual_seed(0)
        loss = InstanceFeatureLoss([64,128,256], [128,256,512], cfg).to(device)
        loss.epoch = epoch
        calls = []
        hooks = [m.register_forward_hook(lambda *x: calls.append(1)) for m in (loss.gate, *loss.adapters.values())]
        s = [torch.randn(2,c,h,h,device=device,requires_grad=True) for c,h in ((64,8),(128,4),(256,2))]
        t = [torch.randn(2,c,h,h,device=device,requires_grad=True) for c,h in ((128,8),(256,4),(512,2))]
        output = loss(s,t)
        output.backward()
        for h in hooks: h.remove()
        assert not calls and all(p.grad is None for p in loss.gate.parameters())
        assert all(p.grad is None for p in loss.adapters.parameters())
        assert all(x.grad is not None and x.grad.abs().sum()>0 for x in s)
        assert all(x.grad is None for x in t)
        assert loss.last_stats["aux_fraction"] == 0 and loss.last_stats["adapter_pretrain"] == 0
    print("PASS: gate and all auxiliary experts inactive, no gradients, epochs 1/31/150",flush=True)
    upstream_test()

if __name__ == "__main__": main()
