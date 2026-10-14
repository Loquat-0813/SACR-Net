import torch
import torch.nn.functional as F
from hsi_shadow.engine import batch_to_device,forward_model

def module_loss(model,out,y):
    ce=F.cross_entropy(out['logits'],y);zero=ce.new_zeros(())
    cr=ca=zero
    if model.supervision!='none':
        cr=F.cross_entropy(out['raw_logits'],y);ca=F.cross_entropy(out['aux_logits'],y)
    auxiliary=.25*(cr+ca)
    return ce+auxiliary,dict(ce=ce,auxiliary=auxiliary,raw_ce=cr,aux_ce=ca)

def coefficients(raw_score,aux_score):
    ratio=raw_score/aux_score.clamp_min(1e-12)
    one=ratio.new_ones(())
    return (1-torch.tanh(ratio),one) if ratio>1 else (one,1-torch.tanh(1/ratio.clamp_min(1e-12)))

def modulate(model,out,y,epoch):
    a=out['logits'].new_ones(());b=a.clone()
    if model.use_ogm and 6<=int(epoch or 0)<=80:
        with torch.no_grad():
            sr=out['raw_logits'].softmax(-1).gather(1,y[:,None]).sum()
            sa=out['aux_logits'].softmax(-1).gather(1,y[:,None]).sum()
            a,b=coefficients(sr,sa)
        for name,p in model.named_parameters():
            if p.grad is None or p.ndim not in (4,5):continue
            if name.startswith('core.spectral_'):p.grad.mul_(a)
            elif name.startswith(('core.quotient_spectral_','extra.')):p.grad.mul_(b)
    return a,b

def train_one_epoch(model,loader,optimizer,device,cfg,epoch=None):
    assert not cfg.get('relight',{}).get('enabled',False)
    assert all(float(cfg.get(k,0))==0 for k in ['consistency_weight','prototype_loss_weight','supcon_loss_weight','gate_l1_weight','label_smoothing'])
    model.train();sums={};total=0
    for i,batch in enumerate(loader):
        if cfg.get('max_train_batches') is not None and i>=cfg['max_train_batches']:break
        batch=batch_to_device(batch,device);optimizer.zero_grad(set_to_none=True)
        out=forward_model(model,batch);loss,parts=module_loss(model,out,batch['y']);loss.backward()
        a,b=modulate(model,out,batch['y'],epoch)
        parts.update(raw_coefficient=a,aux_coefficient=b)
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();n=batch['y'].numel();total+=n
        for k,v in dict(loss=loss,**parts).items():sums[k]=sums.get(k,0)+float(v.detach())*n
    return {k:v/max(total,1) for k,v in sums.items()}
