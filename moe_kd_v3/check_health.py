"""Engineering checks, never an mAP-superiority test and never forces expert balance."""
import argparse,csv,gzip,json,math
from pathlib import Path
from collections import Counter


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [{k.strip():v for k,v in r.items()} for r in csv.DictReader(f)]


def check(run,min_epochs=40):
    root=Path(run);protocol=json.loads((root/'anchor_protocol.json').read_text());cfg=protocol['algorithm']['config']
    metrics=read(root/'results.csv');epochs=len(metrics)
    assert epochs>=min_epochs and [int(r['epoch']) for r in metrics]==list(range(1,epochs+1))
    assert all(math.isfinite(float(v)) for r in metrics for v in r.values())
    losses=read(root/'anchor_losses.csv')
    assert losses and all(math.isfinite(float(v)) for r in losses for v in r.values())
    counts=Counter();choices=Counter();active=0;maxepoch=0;keys=set();last=None;errors=0;expected_per_epoch=None
    with gzip.open(root/'routes.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);ep=int(row['epoch']);key=(ep,int(row['batch']),int(row['sample']))
            if last is not None and key<=last:errors+=1
            last=key;counts[ep]+=1;maxepoch=max(maxepoch,ep)
            assert row['mgd_coefficient']==1. and len(row['mgd_raw_loss_per_scale'])==3
            assert all(math.isfinite(v) for v in row['mgd_raw_loss_per_scale']+row['aux_coefficients']+list(row['probabilities'].values()))
            assert abs(sum(row['probabilities'].values())-1)<1e-5
            assert row['auxiliary'] in ('none','cwd','tafd','at','mimic')
            if row['active']:active+=1;choices[row['auxiliary']]+=1
            if cfg['mode']=='mgd':assert row['auxiliary']=='none' and all(v==0 for v in row['aux_coefficients'])
    assert errors==0 and maxepoch==epochs
    if protocol['route_every']==1:
        assert all(counts[e]==counts[1] for e in range(1,epochs+1)),'Missing/duplicate sample routes'
        expected_per_epoch=counts[1]
    norms=read(root/'norm_calibration.csv') if (root/'norm_calibration.csv').exists() else []
    if cfg['mode']!='mgd' and cfg['aux_fraction']>0 and epochs>cfg['warmup_epochs']:
        assert active>0 and norms,'Residual never opened/calibrated'
        assert all(math.isfinite(float(r[k])) for r in norms for k in ['mgd_norm','aux_raw_norm','scale_factor'])
    probes=read(root/'task_probes.csv') if (root/'task_probes.csv').exists() else []
    if cfg['mode']=='gated' and cfg['task_coef']>0 and active:
        assert probes and any(float(r['task_ce'])>0 for r in losses),'Gate supervision never ran'
        assert all(math.isfinite(float(r[k])) for r in probes for k in ['base_query_loss','candidate_query_loss','relative_gain','predicted_relative_gain','query_alignment'])
    targets=Counter(r['target'] for r in probes);positive=sum(r['eligible']=='True' for r in probes)
    warnings=[]
    if probes and positive==0:warnings.append('No positive-margin virtual gains; none is valid, but adaptive benefit is unproven.')
    if active and choices.get('none',0)/active>.95:warnings.append('More than 95% residual-off; inspect gains before claiming expert adaptation.')
    report={'epochs':epochs,'mode':cfg['mode'],'pass':True,'sample_routes':sum(counts.values()),
            'logged_samples_per_epoch':expected_per_epoch,'route_errors':errors,'mgd_coefficient':1.,
            'active_routes':active,'active_choices':dict(choices),'probe_rows':len(probes),'target_row_counts':dict(targets),
            'positive_margin_candidate_fraction':positive/max(len(probes),1),'warnings':warnings,
            'positive_target_fraction':sum(r['target']!='none' for r in probes)/max(len(probes),1),
            'meaning':'Engineering guard only; no assertion of mAP gain, optimal selection, or exact resumed replay.'}
    (root/'v3_health.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--min-epochs',type=int,default=40)
    a=p.parse_args();check(a.run,a.min_epochs)
