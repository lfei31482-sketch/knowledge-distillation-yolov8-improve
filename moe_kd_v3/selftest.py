"""Numerical and state-isolation tests; no VOC data or teacher weights required."""
import copy,unittest,tempfile,json,gzip,csv,os
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[1]/'.moe_yolo_config'))
os.environ['VOC_OFFLINE_TRAINING']='1'
import torch
from torch import nn
from .loss import AnchorConfig,AnchorFeatureLoss,RawMGD,AUX
from .lookahead import virtual_loss,virtual_loss_and_gradient,marginal_gain,choice_target


class AnchorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.sc=[4,8,16];self.tc=[8,16,32]
        self.s=[torch.randn(4,c,h,h,requires_grad=True) for c,h in [(4,8),(8,4),(16,2)]]
        self.t=[torch.randn(4,c,h,h,requires_grad=True) for c,h in [(8,8),(16,4),(32,2)]]

    def make(self,**kwargs):return AnchorFeatureLoss(self.sc,self.tc,AnchorConfig(**kwargs))

    def test_raw_mgd_matches_original_loss_and_student_gradients(self):
        from ultralytics.utils.distill_loss import FeatureLoss
        raw=RawMGD(self.sc,self.tc)
        native=FeatureLoss(self.sc,self.tc,distiller='mgd').cpu()
        native.align_module.load_state_dict(raw.aligner.align_module.state_dict())
        native.norm.load_state_dict(raw.aligner.norm.state_dict())
        native.feature_loss.generation.load_state_dict(raw.generators.state_dict())
        s2=[x.detach().clone().requires_grad_() for x in self.s]
        torch.manual_seed(101);actual=sum(v.mean() for v in raw(self.s,self.t))
        torch.manual_seed(101);expected=native(s2,[t.detach() for t in self.t])
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,[*self.s,*raw.parameters()])
        ge=torch.autograd.grad(expected,[*s2,*native.parameters()])
        for a,b in zip(ga,ge):torch.testing.assert_close(a,b)

    def test_zero_aux_is_exact_raw_mgd(self):
        m=self.make(mode='gated',warmup_epochs=0,aux_fraction=0)
        other=copy.deepcopy(m.mgd)
        torch.manual_seed(9);actual=m(self.s,self.t)
        torch.manual_seed(9);expected=sum(v.mean() for v in other(self.s,self.t))
        torch.testing.assert_close(actual,expected)
        self.assertTrue((m.trace['ids']==0).all())
        actual.backward();self.assertTrue(all(t.grad is None for t in self.t))

    def test_shadow_warmup_never_changes_student_gradient(self):
        m=self.make(warmup_epochs=3);other=copy.deepcopy(m.mgd)
        torch.manual_seed(10);actual=m(self.s,self.t)
        torch.manual_seed(10);expected=sum(v.mean() for v in other(self.s,self.t))
        for a,b in zip(torch.autograd.grad(actual,self.s,retain_graph=True),torch.autograd.grad(expected,self.s)):
            torch.testing.assert_close(a,b)
        actual.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in m.adapters.parameters() if p.grad is not None),0)
        self.assertTrue(all(t.grad is None for t in self.t))

    def test_every_image_always_receives_mgd(self):
        m=self.make(mode='fixed',expert='cwd',warmup_epochs=0)
        m.scale_updates.fill_(1);m.aux_scales.fill_(.1)
        calls=[];hooks=[g.register_forward_hook(lambda *x:calls.append(x[-1].shape[0])) for g in m.mgd.generators]
        m(self.s,self.t).backward()
        for h in hooks:h.remove()
        self.assertEqual(calls,[4,4,4]);self.assertTrue((m.trace['ids']==1).all())
        self.assertGreater(m.last_stats['aux_weighted'],0)

    def test_primary_auxiliary_parameter_sets_disjoint(self):
        m=self.make();primary={id(p) for p in m.mgd.parameters()}
        self.assertFalse(primary&{id(p) for p in m.adapters.parameters()})
        groups=[{id(p) for p in m.adapters[e].parameters()} for e in AUX]
        self.assertTrue(all(not a&b for i,a in enumerate(groups) for b in groups[i+1:]))

    def test_auxiliary_construction_does_not_change_mgd_rng(self):
        torch.manual_seed(123);raw=RawMGD(self.sc,self.tc);r1=torch.get_rng_state()
        torch.manual_seed(123);anchor=self.make();r2=torch.get_rng_state()
        self.assertTrue(torch.equal(r1,r2))
        for a,b in zip(raw.parameters(),anchor.mgd.parameters()):torch.testing.assert_close(a,b)

    def test_router_rng_is_separate_and_resumable(self):
        m=self.make();p=torch.ones(40,5)/5;before=torch.get_rng_state()
        m._sample(p);state=m.router_state();reference=m._sample(p)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        other=self.make();other.set_router_state(state)
        self.assertTrue(torch.equal(reference,other._sample(p)))

    def test_gate_has_no_raw_kd_reward(self):
        m=self.make(mode='gated',warmup_epochs=0);m.scale_updates.fill_(1)
        m(self.s,self.t).backward()
        self.assertTrue(all(p.grad is None or not torch.count_nonzero(p.grad) for p in m.gate.parameters()))

    def test_gate_supervision_backpropagates(self):
        m=self.make(warmup_epochs=0);m.scale_updates.fill_(1)
        m.task_supervisor=lambda s,t,b,p:-p[:,1].log().mean()
        m(self.s,self.t).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in m.gate.parameters() if p.grad is not None),0)

    def test_margin_allows_none(self):
        self.assertEqual(choice_target([-.1,-.2,0.,-.4],1e-5),0)
        self.assertEqual(choice_target([1e-7,2e-7,0.,0.],1e-5),0)
        self.assertEqual(choice_target([.001,.003,-.01,.002],1e-5),2)

    def test_virtual_step_matches_manual_linear_update(self):
        m=nn.Linear(2,1,bias=False);m.weight.data.copy_(torch.tensor([[1.,2.]]))
        q={'img':torch.tensor([[2.,-1.]])};criterion=lambda pred,batch:((pred-1).square().mean(),None)
        g={'weight':torch.tensor([[.5,.25]])};a={'weight':torch.tensor([[.1,-.1]])}
        value=virtual_loss(m,criterion,q,g,a,step=.1)
        expected=((q['img']@(m.weight.detach()-.1*(g['weight']+a['weight'])).T-1)**2).mean().item()
        self.assertAlmostEqual(value,expected,places=6)

    def test_virtual_step_restores_bn_hooks_flags_parameters_and_gradients(self):
        m=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.ReLU(),nn.Conv2d(4,1,1)).train()
        before={k:v.clone() for k,v in m.state_dict().items()};flags=[x.training for x in m.modules()]
        events=[];handle=m[0].register_forward_hook(lambda *args:events.append(1))
        for p in m.parameters():p.grad=torch.ones_like(p)
        query={'img':torch.randn(2,3,8,8)};grads={k:torch.randn_like(p)*.01 for k,p in m.named_parameters()}
        criterion=lambda pred,batch:(pred.square().mean(),None)
        virtual_loss(m,criterion,query,grads)
        self.assertEqual(events,[]);self.assertEqual(flags,[x.training for x in m.modules()])
        for k,v in m.state_dict().items():torch.testing.assert_close(before[k],v)
        self.assertTrue(all(torch.equal(p.grad,torch.ones_like(p)) for p in m.parameters()))
        m(query['img']);self.assertEqual(events,[1]);handle.remove()

    def test_resume_reconciles_interrupted_csv_and_gzip(self):
        from .trainer import reconcile_logs
        with tempfile.TemporaryDirectory(dir=Path.cwd().resolve()) as directory:
            p=Path(directory)
            assert p.resolve().is_relative_to(Path.cwd().resolve())
            (p/'results.csv').write_text('  epoch, value\n1,1\n2,2\n3,3\n')
            with gzip.open(p/'routes.jsonl.gz','wt') as f:
                for epoch in [1,2,3]:f.write(json.dumps({'epoch':epoch})+'\n')
            reconcile_logs(p,2)
            with (p/'results.csv').open() as stream:self.assertEqual(len(list(csv.DictReader(stream))),2)
            with gzip.open(p/'routes.jsonl.gz','rt') as f:self.assertEqual([json.loads(r)['epoch'] for r in f],[1,2])
            self.assertEqual(len(list(p.glob('*.interrupted_*'))),2)

    def test_bad_settings_rejected(self):
        for kw in [{'aux_fraction':1.},{'virtual_lr':0},{'exploration':1.},{'probe_samples':0}]:
            with self.assertRaises(ValueError):AnchorConfig(**kw).validate()

    def test_marginal_gain_sign_matches_real_small_step(self):
        m=nn.Linear(2,1,bias=False);m.weight.data.copy_(torch.tensor([[.3,-.2]]))
        query={'img':torch.tensor([[1.,2.]])};criterion=lambda pred,batch:((pred-.8).square().mean(),None)
        base={'weight':torch.tensor([[.1,.1]])};aux={'weight':torch.tensor([[-.2,-.4]])}
        before=m.weight.detach().clone()
        value,qg=virtual_loss_and_gradient(m,criterion,query,base,step=.001)
        estimate,alignment=marginal_gain(base,aux,qg,value,step=.001)
        actual=(value-virtual_loss(m,criterion,query,base,aux,step=.001))/value
        self.assertGreater(alignment,.9);self.assertGreater(estimate,0.)
        self.assertAlmostEqual(estimate,actual,places=5)
        torch.testing.assert_close(m.weight,before);self.assertIsNone(m.weight.grad)


if __name__=='__main__':unittest.main(verbosity=2)
