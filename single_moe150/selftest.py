"""Synthetic tests; never loads real checkpoints."""
import sys
import torch
from moe_kd_v3.loss import AnchorConfig, AnchorFeatureLoss
from instance_moe_v5.loss import InstanceFeatureLoss
from single_moe150.trainer import SingleConfig
from instance_moe_response_v1.selftest import run as response_test

def main():
    for seed in (0,41):
        torch.manual_seed(seed)
        a=AnchorFeatureLoss([64,128,256],[128,256,512],AnchorConfig(mode='mgd'))
        torch.manual_seed(seed)
        b=InstanceFeatureLoss([64,128,256],[128,256,512],SingleConfig(learning_rate=.01,warmup_epochs=30))
        assert all(torch.equal(a.mgd.state_dict()[k],v) for k,v in b.mgd.state_dict().items())
    try:
        SingleConfig(stage1_path='not_allowed',stage1_sha256='a'*64).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('Prior stage state was accepted')
    print('PASS: seeded MGD initialization matches baseline; stage-one load rejected',flush=True)
    response_test(sys.argv[1] if len(sys.argv)>1 else 'cpu')

if __name__=='__main__': main()
