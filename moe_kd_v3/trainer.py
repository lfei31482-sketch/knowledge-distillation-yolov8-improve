"""Additive YOLO adapter, full raw MGD plus a separately controlled residual."""
import copy,csv,gzip,hashlib,json,os,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .yolo_distiller import DetectionDistiller
from ultralytics.utils.torch_utils import init_seeds
from .loss import AUX,CHOICES,AnchorConfig,AnchorFeatureLoss
from .lookahead import virtual_loss,virtual_loss_and_gradient,marginal_gain,choice_target


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def addon_hash():
    h=hashlib.sha256();root=Path(__file__).resolve().parent
    for name in ('loss.py','trainer.py','lookahead.py','yolo_distiller.py'):
        h.update(name.encode());h.update((root/name).read_bytes())
    h.update((root.parent/'train_moe_v3.py').read_bytes())
    return h.hexdigest()


def atomic_save(payload,path):
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp');torch.save(payload,temp);os.replace(temp,p)


def append_csv(path,rows):
    if not rows:return
    path=Path(path);exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));
        if not exists:w.writeheader()
        w.writerows(rows)


def query_slice(batch,size):
    size=min(size,batch['img'].shape[0]);mask=batch['batch_idx'].view(-1)<size
    return {'img':batch['img'][:size].detach().clone(),
            'batch_idx':batch['batch_idx'][mask].detach().clone(),
            'cls':batch['cls'][mask].detach().clone(),'bboxes':batch['bboxes'][mask].detach().clone(),
            'im_file':list(batch['im_file'][:size])}


def reconcile_logs(root,completed):
    """Discard uncheckpointed tail from active logs, preserving originals as backups."""
    root=Path(root);stamp=time.strftime('%Y%m%d_%H%M%S');clean=root/'LOGS_CLEAN.json'
    if clean.exists() and json.loads(clean.read_text())['epoch']==completed:return
    for p in [*root.glob('*.csv'),root/'routes.jsonl.gz']:
        if not p.exists():continue
        temporary=p.with_name(p.name+'.recovered');backup=p.with_name(p.name+'.interrupted_'+stamp)
        if backup.exists():raise FileExistsError(backup)
        if p.name.endswith('.gz'):
            with gzip.open(temporary,'wt',encoding='utf-8') as dst:
                try:
                    with gzip.open(p,'rt',encoding='utf-8') as src:
                        for line in src:
                            row=json.loads(line)
                            if int(row['epoch'])>completed:break
                            dst.write(line)
                except (EOFError,OSError,json.JSONDecodeError):
                    # A killed gzip writer can leave the final, unsaved member incomplete.
                    # Subsequent health checks verify the committed prefix's row counts.
                    pass
        else:
            with p.open(encoding='utf-8-sig',newline='') as src,temporary.open('w',encoding='utf-8',newline='') as dst:
                reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames);writer.writeheader()
                for row in reader:
                    try:ep=int(float(next((v for k,v in row.items() if k and k.strip()=='epoch'),'').strip()))
                    except (ValueError,AttributeError):break
                    if ep>completed:break
                    writer.writerow(row)
        os.replace(p,backup);os.replace(temporary,p)
        print('Preserved interrupted log:',backup,flush=True)


class SeededMixin:
    def get_model(self,*args,**kwargs):
        init_seeds(int(self.args.seed),deterministic=self.args.deterministic)
        return super().get_model(*args,**kwargs)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        loader=super().get_dataloader(dataset_path,batch_size,rank,mode)
        if mode=='train':
            gen=getattr(loader,'generator',None)
            if gen is not None:gen.manual_seed(6148914691236517205+int(self.args.seed)+rank)
            if hasattr(loader,'reset'):loader.reset()
        return loader


class CaptureCriterion:
    def __init__(self,criterion):self.criterion,self.last=criterion,None
    def __call__(self,*args,**kwargs):
        result=self.criterion(*args,**kwargs);self.last=result[0].sum();return result


class AnchorDistiller(SeededMixin,DetectionDistiller):
    def __init__(self,config=None,resume_state=None,route_every=1,diagnostic_every=200,**kwargs):
        self.anchor_config=config or AnchorConfig();self.anchor_config.validate()
        self.resume_state=Path(resume_state).resolve() if resume_state else None
        self.route_every,self.diagnostic_every=route_every,diagnostic_every
        self.batch_index=-1;self.current_batch=None;self.capture=None;self.route_stream=None
        self.query_buffer=None;self._last_saved_epoch=-1
        super().__init__(**kwargs)
        self.teacher_sha=file_hash(self.args.teacher_weights)
        self.callbacks['on_train_epoch_start'].append(self._epoch_start)
        self.callbacks['on_train_batch_start'].append(self._batch_start)
        self.callbacks['on_train_batch_end'].append(self._batch_end)

    def setup_teacher_model(self):
        ckpt=super().setup_teacher_model()
        if self.teacher_model.model[-1].nc!=self.data['nc']:raise ValueError('Teacher classes differ')
        return ckpt

    def build_optimizer(self,model,**kwargs):
        old=kwargs.pop('feature_model');tc=[n.num_features for n in old.norm]
        sc=[a.in_channels if isinstance(a,torch.nn.Conv2d) else t for a,t in zip(old.align_module,tc)]
        if sc!=[64,128,256] or tc!=[128,256,512]:raise ValueError(f'Expected vanilla YOLOv8n/s: {sc}, {tc}')
        self.kd_feature_loss=AnchorFeatureLoss(sc,tc,self.anchor_config).to(self.device)
        opt=super().build_optimizer(model,feature_model=self.kd_feature_loss,**kwargs)
        gate=list(self.kd_feature_loss.gate.parameters());ids={id(p) for p in gate}
        for group in opt.param_groups:group['params']=[p for p in group['params'] if id(p) not in ids]
        opt.add_param_group({'params':gate,'lr':kwargs['lr']*self.anchor_config.gate_lr_scale,
                             'weight_decay':0.,'anchor_gate_group':True})
        return opt

    def _setup_train(self,world_size):
        if world_size not in (0,1):raise ValueError('One GPU/process only')
        super()._setup_train(world_size)
        if self.amp:raise ValueError('V3 matched protocol requires FP32')
        self.capture=CaptureCriterion(self.model.criterion);self.model.criterion=self.capture
        self.query_criterion=copy.deepcopy(self.capture.criterion)
        self.kd_feature_loss.task_supervisor=self._task_supervise
        self.kd_feature_loss.observer=self._diagnose
        if self.resume_state:self._restore()
        clean=self.save_dir/'LOGS_CLEAN.json'
        if clean.exists():clean.unlink()
        self.route_stream=gzip.open(self.save_dir/'routes.jsonl.gz','at',encoding='utf-8')
        protocol={'algorithm':self.kd_feature_loss.metadata(),'seed':self.args.seed,
                  'teacher_sha256':self.teacher_sha,'teacher':str(self.args.teacher_weights),
                  'addon_sha256':addon_hash(),'data_yaml_sha256':file_hash(self.args.data),'precision':'FP32',
                  'route_every':self.route_every,'diagnostic_every':self.diagnostic_every,
                  'resume':str(self.resume_state) if self.resume_state else None,
                  'student_auxiliary_modules':'none; all training auxiliaries are separately owned',
                  'query_source':'previous augmented TRAINING batch, no validation/test feedback',
                  'virtual_update':'one plain SGD step, student-only, norm clip 10; no momentum/accumulation',
                  'gate_target':'marginal query-gradient gain after virtual MGD base step, checked against actual finite differences',
                  'norm_budget':'per-scale full-batch feature-gradient EMA, not a global parameter-gradient bound',
                  'torch':torch.__version__}
        (self.save_dir/'anchor_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

    def _epoch_start(self,trainer):
        self.batch_index=-1;self.kd_feature_loss.epoch=self.epoch

    def _batch_start(self,trainer):self.batch_index+=1

    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch);self.current_batch=batch;return batch

    def _task_supervise(self,features,teacher,base_vectors,probs):
        c=self.anchor_config;m=self.kd_feature_loss;zero=probs.new_zeros(())
        if self.batch_index%c.probe_every and m.scale_updates.item()>0:return zero
        started=time.perf_counter();n=probs.shape[0]
        # Always calibrate all residuals on the SAME batch and current raw MGD gradients.
        all_aux={name:m.aux_vectors(name,features,teacher) for name in AUX}
        normrows=[]
        for l,f in enumerate(features):
            gm=torch.autograd.grad(base_vectors[l].mean(),f,retain_graph=True,allow_unused=True)[0]
            nm=gm.detach().float().norm() if gm is not None else f.new_zeros(())
            for e,name in enumerate(AUX):
                ga=torch.autograd.grad(all_aux[name][l].mean(),f,retain_graph=True,allow_unused=True)[0]
                na=ga.detach().float().norm() if ga is not None else f.new_zeros(())
                desired=(nm/na.clamp_min(1e-12)).clamp(min=1e-8,max=c.max_scale)
                if not torch.isfinite(desired):raise FloatingPointError('Invalid auxiliary gradient scale')
                with torch.no_grad():
                    if m.scale_updates.item()==0:m.aux_scales[l,e].copy_(desired)
                    else:m.aux_scales[l,e].lerp_(desired,c.scale_ema)
                normrows.append({'epoch':self.epoch+1,'batch':self.batch_index,'scale':f'P{l+3}',
                                 'expert':name,'mgd_norm':float(nm),'aux_raw_norm':float(na),
                                 'scale_factor':float(m.aux_scales[l,e]),'target_fraction':c.aux_fraction})
        m.scale_updates.add_(1)
        append_csv(self.save_dir/'norm_calibration.csv',normrows)
        if c.mode!='gated' or c.task_coef==0 or self.query_buffer is None:return zero
        params=[(name,p) for name,p in self.model.named_parameters() if p.requires_grad]
        base_loss=self.capture.last+sum(v.mean() for v in base_vectors)*self.args.feature_loss_ratio
        gg=torch.autograd.grad(base_loss,[p for _,p in params],retain_graph=True,allow_unused=True)
        base_g={name:g.detach() if g is not None else None for (name,_),g in zip(params,gg)}
        query=self.query_buffer
        base_query,query_g=virtual_loss_and_gradient(self.model,self.query_criterion,query,base_g,step=c.virtual_lr)
        indices=[(self.epoch*17+(self.batch_index//c.probe_every)*c.probe_samples+j)%n
                 for j in range(min(c.probe_samples,n))]
        targets=[];rows=[]
        for i in indices:
            gains=[];candidate_losses=[];predicted=[];alignments=[];eligible=[]
            for e,name in enumerate(AUX):
                # Evaluate the full configured budget, even while the live ramp is small.
                item=sum(all_aux[name][l][i]*m.aux_scales[l,e].detach()*c.aux_fraction
                         for l in range(3))*self.args.feature_loss_ratio/n
                ag=torch.autograd.grad(item,[p for _,p in params],retain_graph=True,allow_unused=True)
                aux_g={name:g.detach() if g is not None else None for (name,_),g in zip(params,ag)}
                value=virtual_loss(self.model,self.query_criterion,query,base_g,aux_g,step=c.virtual_lr)
                gain=(base_query-value)/max(abs(base_query),1e-8)
                gains.append(gain);candidate_losses.append(value)
                estimate,alignment=marginal_gain(base_g,aux_g,query_g,base_query,c.virtual_lr)
                predicted.append(estimate);alignments.append(alignment)
                # FP32 finite differences close to zero are not reliable sign tests.
                # Reject a clearly adverse actual difference; tolerate scalar roundoff.
                reliable=(alignment>c.min_alignment and estimate>c.min_gain
                          and gain>=-max(5e-7,abs(estimate)*.5))
                eligible.append(reliable)
            target=choice_target([g if ok else -float('inf') for g,ok in zip(predicted,eligible)],c.min_gain)
            targets.append(target)
            for e,name in enumerate(AUX):
                rows.append({'epoch':self.epoch+1,'batch':self.batch_index,'sample':i,
                             'anchor':str(self.current_batch['im_file'][i]),'expert':name,
                             'base_query_loss':base_query,'candidate_query_loss':candidate_losses[e],
                             'relative_gain':gains[e],'minimum_gain':c.min_gain,'target':CHOICES[target],
                             'predicted_relative_gain':predicted[e],'query_alignment':alignments[e],
                             'minimum_alignment':c.min_alignment,'eligible':eligible[e],
                             'probability':float(probs[i,e+1].detach()),
                             'query_anchors':json.dumps(query['im_file'],ensure_ascii=False),
                             'virtual_lr':c.virtual_lr,'probe_seconds':time.perf_counter()-started})
        append_csv(self.save_dir/'task_probes.csv',rows)
        selected=probs[indices];target_tensor=torch.tensor(targets,device=probs.device)
        # Hard positive-margin decisions replace the almost-uniform V2 soft labels.
        loss=torch.nn.functional.nll_loss(selected.clamp_min(1e-12).log(),target_tensor)
        print(f'\n[V3 probe] epoch={self.epoch+1} batch={self.batch_index} '
              f'targets={[CHOICES[t] for t in targets]} base_query={base_query:.5f} '
              f'predicted_gain_range=[{min(r["predicted_relative_gain"] for r in rows):.3g},'
              f'{max(r["predicted_relative_gain"] for r in rows):.3g}] seconds={time.perf_counter()-started:.2f}',flush=True)
        return loss

    def _diagnose(self,trace):
        if not self.diagnostic_every or self.batch_index%self.diagnostic_every:return
        features=trace['features'];n=features[0].shape[0];lam=self.args.feature_loss_ratio
        gd=torch.autograd.grad(self.capture.last,features,retain_graph=True,allow_unused=True)
        gb=torch.autograd.grad(self.capture.last+trace['base_loss']*lam,features,retain_graph=True,allow_unused=True)
        rows=[]
        for l,f in enumerate(features):
            if gd[l] is None:continue
            mg=torch.autograd.grad(trace['base_vectors'][l][0]*lam/n,f,retain_graph=True)[0][0].detach().flatten()
            branches=[('mgd',mg,1.)]
            for scale,e,ix,raw,weighted in trace['records']:
                if scale!=l:continue
                at=(ix==0).nonzero(as_tuple=True)[0]
                if at.numel():
                    ga=torch.autograd.grad(weighted[at[0]]*lam/n,f,retain_graph=True)[0][0].detach().flatten()
                    coefficient=trace['fraction']*float(self.kd_feature_loss.aux_scales[l,e])
                    branches.append((AUX[e],ga,coefficient))
            for branch,g,coefficient in branches:
                d=gd[l][0].detach().flatten();b=gb[l][0].detach().flatten()
                cosine=lambda x,y:float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-20))
                rows.append({'epoch':self.epoch+1,'batch':self.batch_index,'sample':0,
                             'anchor':str(self.current_batch['im_file'][0]),'scale':f'P{l+3}',
                             'branch':branch,'cosine_vs_detection':cosine(g,d),'cosine_vs_base':cosine(g,b),
                             'norm_ratio_vs_mgd':float(g.norm()/mg.norm().clamp_min(1e-12)),
                             'coefficient':coefficient})
        append_csv(self.save_dir/'expert_gradients.csv',rows)

    def _batch_end(self,trainer):
        m=self.kd_feature_loss;trace=m.trace
        # YOLO invokes this after backward; gradient diagnostics must run BEFORE backward.
        if self.route_every and self.batch_index%self.route_every==0:
            ids=trace['ids'].cpu().tolist();pp=trace['probs'].cpu().tolist();batch=self.current_batch
            targetidx=batch['batch_idx'].view(-1);boxes=batch['bboxes']
            for i,anchor in enumerate(batch['im_file']):
                bb=boxes[targetidx==i];areas=(bb[:,2]*bb[:,3]).detach().cpu().tolist()
                selected=ids[i];name=CHOICES[selected]
                coefficients=[trace['fraction']*float(m.aux_scales[l,selected-1]) if selected else 0. for l in range(3)]
                row={'epoch':self.epoch+1,'batch':self.batch_index,'sample':i,'anchor':str(anchor),
                     'unit':'augmented_training_tensor_anchor_not_all_mosaic_sources',
                     'objects':len(areas),'relative_areas':areas,'mgd_coefficient':1.,
                     'mgd_raw_loss_per_scale':[float(v[i].detach()) for v in trace['base_vectors']],
                     'auxiliary':name,'aux_coefficients':coefficients,
                     'probabilities':dict(zip(CHOICES,pp[i])),'active':trace['active']}
                self.route_stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            self.route_stream.flush()
        append_csv(self.save_dir/'anchor_losses.csv',[{'epoch':self.epoch+1,'batch':self.batch_index,**m.last_stats}])
        if self.batch_index%50==0:
            print(f'\n[V3 {self.anchor_config.mode} epoch={self.epoch+1} batch={self.batch_index}] '
                  f'{m.last_stats} MGD_coefficient=1',flush=True)
        self.query_buffer=query_slice(self.current_batch,self.anchor_config.probe_samples)
        m.trace={};m.last_scale_losses=();self.capture.last=None

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),10.,error_if_nonfinite=True)
        # Auxiliary/gate norms cannot change the MGD generator/adapter clipping factor.
        m=self.kd_feature_loss
        torch.nn.utils.clip_grad_norm_(m.mgd.parameters(),10.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(m.adapters.parameters(),10.,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(m.gate.parameters(),1.,error_if_nonfinite=True)
        self.scaler.step(self.optimizer);self.scaler.update();self.optimizer.zero_grad(set_to_none=True)
        if self.ema:self.ema.update(self.model)

    def save_model(self):
        self.model.criterion=self.capture.criterion
        try:super().save_model()
        finally:self.model.criterion=self.capture
        query={k:v.cpu() if torch.is_tensor(v) else v for k,v in self.query_buffer.items()} if self.query_buffer else None
        state={'format':3,'epoch':self.epoch,'config':asdict(self.anchor_config),'addon_sha256':addon_hash(),
               'data_yaml_sha256':file_hash(self.args.data),'teacher_sha256':self.teacher_sha,'args':vars(self.args),
               'student':self.model.state_dict(),'ema':self.ema.ema.state_dict(),'ema_updates':self.ema.updates,
               'auxiliary':self.kd_feature_loss.state_dict(),'router_rng':self.kd_feature_loss.router_state(),
               'query_buffer':query,'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),
               'scaler':self.scaler.state_dict(),'best_fitness':self.best_fitness,'python_rng':random.getstate(),
               'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        atomic_save(state,self.wdir/'moe_last.pt')
        if self.best_fitness==self.fitness:atomic_save(state,self.wdir/'moe_best.pt')
        self._last_saved_epoch=self.epoch

    def _restore(self):
        s=torch.load(self.resume_state,map_location='cpu',weights_only=False)
        if s.get('format')!=3:raise ValueError('Only V3 training checkpoints can resume V3')
        if s['addon_sha256']!=addon_hash() or s['config']!=asdict(self.anchor_config):raise ValueError('Source/config changed')
        if s['teacher_sha256']!=self.teacher_sha or s['data_yaml_sha256']!=file_hash(self.args.data):raise ValueError('Teacher/data changed')
        keys=('seed','batch','imgsz','epochs','lr0','lrf','data','feature_loss_ratio','model','fraction','nbs',
              'mosaic','warmup_epochs','close_mosaic','optimizer','momentum','weight_decay','workers')
        for key in keys:
            if str(s['args'][key])!=str(getattr(self.args,key)):raise ValueError('Resume argument changed: '+key)
        self.model.load_state_dict(s['student']);self.ema.ema.load_state_dict(s['ema']);self.ema.updates=s['ema_updates']
        self.kd_feature_loss.load_state_dict(s['auxiliary']);self.kd_feature_loss.set_router_state(s['router_rng'])
        self.optimizer.load_state_dict(s['optimizer']);self.scheduler.load_state_dict(s['scheduler']);self.scaler.load_state_dict(s['scaler'])
        self.start_epoch=s['epoch']+1;self._last_saved_epoch=s['epoch'];self.best_fitness=s['best_fitness']
        if self.start_epoch>=self.epochs:raise ValueError('Training already complete')
        if s['query_buffer']:self.query_buffer={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in s['query_buffer'].items()}
        reconcile_logs(self.save_dir,self.start_epoch)
        random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'])
        if s['cuda_rng'] is not None:torch.cuda.set_rng_state_all(s['cuda_rng'])
        print(f'Resumed V3 at epoch {self.start_epoch+1}; dataloader prefetch is not bit-exact.',flush=True)

    def distill(self):
        try:super().distill()
        finally:
            if self.route_stream:
                self.route_stream.close()
                if self._last_saved_epoch==getattr(self,'epoch',-2):
                    (self.save_dir/'LOGS_CLEAN.json').write_text(json.dumps({'epoch':self.epoch+1}),encoding='utf-8')
