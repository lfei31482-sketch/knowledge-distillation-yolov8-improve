"""CPU tests with synthetic logits and boxes; no checkpoint access."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import torch
from instance_moe_response_v1.response import ResponseLoss


def run(device='cpu'):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = SimpleNamespace(epoch=30,batch_index=1,save_dir=Path(tmp))
        loss_fn = ResponseLoss(trainer,conf_min=.25,iou_min=.30)
        teacher=[]
        for size in (8,4,2):
            # Regression predicts roughly one feature cell per side.
            x=torch.full((1,84,size,size),-10.,device=device)
            for edge in range(4): x[:,edge*16+1]=10.
            x[:,64:]=-10.; x[:,64]=10.
            teacher.append(x.requires_grad_())
        batch={'img':torch.zeros(1,3,64,64,device=device),'batch_idx':torch.tensor([0]),
               'cls':torch.tensor([0]),'bboxes':torch.tensor([[.5,.5,.5,.5]])}
        same=[x.detach().clone().requires_grad_() for x in teacher]
        result=loss_fn(same,teacher,batch)
        assert result.abs()<1e-6, result
        result.backward()
        assert all(x.grad is None for x in teacher)
        different=[x.detach().clone().requires_grad_() for x in teacher]
        with torch.no_grad():
            for x in different:
                x[:,64]-=2.; x[:,1]-=2.
        loss=loss_fn(different,teacher,batch)
        assert torch.isfinite(loss) and loss.item()>0,loss
        loss.backward()
        assert any(x.grad is not None and x.grad.abs().sum()>0 for x in different)
        assert all(x.grad is None for x in teacher)
        empty={**batch,'batch_idx':torch.empty(0,dtype=torch.long),
               'cls':torch.empty(0),'bboxes':torch.empty(0,4)}
        zeros=[x.detach().clone().requires_grad_() for x in teacher]
        z=loss_fn(zeros,teacher,empty)
        assert z.item()==0; z.backward()
        trainer.epoch=19
        assert loss_fn(different,teacher,batch).item()==0
        trainer.epoch=20
        assert 0 < loss_fn(different,teacher,batch).item() < loss.item()
        broken=[x.detach().clone() for x in teacher]; broken[0]=broken[0][:,:,:-1]
        try: loss_fn(different,broken,batch)
        except ValueError: pass
        else: raise AssertionError('shape mismatch accepted')
        trainer.epoch=30
        for scale, width in enumerate((.25,.50,1.0)):
            s=[x.detach().clone().requires_grad_() for x in teacher]
            with torch.no_grad(): s[scale][:,64]-=2.
            box={**batch, 'bboxes':torch.tensor([[.5,.5,width,width]])}
            value=loss_fn(s,teacher,box); value.backward()
            assert s[scale].grad is not None and s[scale].grad.abs().sum()>0, scale
    print(f'PASS {device}: response KL equality, per-scale gradients, teacher detach, empty GT, warmup and shape guard')

if __name__=='__main__': run(sys.argv[1] if len(sys.argv)>1 else 'cpu')
