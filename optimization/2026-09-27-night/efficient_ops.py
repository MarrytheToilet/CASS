"""Mathematically equivalent extraction and compiled gated steering kernels."""
import random
import numpy as np
import torch
from cass.tasks import build_fewshot_pair_prompts


@torch.no_grad()
def selected_hiddens(hlm,prompts,layers,batch_size=48):
    assert all(0<l<hlm.L for l in layers)
    collected={l:[] for l in layers}
    for start in range(0,len(prompts),batch_size):
        enc=hlm.tok(prompts[start:start+batch_size],return_tensors='pt',padding=True).to(hlm.device)
        handles=[]; current={}
        for l in layers:
            def hook(module,inputs,output,layer=l):
                h=output[0] if isinstance(output,tuple) else output
                current[layer]=h[:,-1].detach().clone()
            handles.append(hlm.layers[l-1].register_forward_hook(hook))
        try:
            # Activations need the decoder, not the unused vocabulary projection.
            hlm.model.model(**enc,use_cache=False,output_hidden_states=False)
        finally:
            for handle in handles:handle.remove()
        for l in layers:collected[l].append(current[l].float().cpu())
    return {l:torch.cat(v) for l,v in collected.items()}


def extract_selected(hlm,examples,layers,seed=0,batch_size=48):
    clean,corr=build_fewshot_pair_prompts(examples,random.Random(9000+seed),n_reps=6)
    H=selected_hiddens(hlm,clean+corr,layers,batch_size=batch_size)
    half=len(clean)
    return {l:(h[:half]-h[half:]).reshape(len(examples),6,-1).mean(1) for l,h in H.items()}


def selected_zlist(D,Z):
    return [D.project_out_shared(D.stack({l:Z[l][i].numpy() for l in D.layers}))
            for i in range(len(next(iter(Z.values()))))]


def gated_kernel(h,Q,delta,mu,gate):
    h=h.float()
    diff=mu-h
    proj=(diff@Q)@Q.T
    alpha=(2*(diff-proj).norm(dim=1)/(h.norm(dim=1)+1e-8)).clamp(max=1).unsqueeze(1)
    return h+(gate*alpha+(1-gate))*delta+gate*alpha*proj


compiled_kernel=torch.compile(gated_kernel,fullgraph=True,dynamic=True)


def compiled_ops(D,code,z):
    weights=np.array([np.linalg.norm(code.coeffs[n]) for n in code.support])
    if not len(weights):
        from cass.pipeline import ops_for
        return ops_for(D,code,delta_vec=z)
    weights/=weights.sum()
    target=sum(w*np.linalg.norm(D.anchors[n]) for w,n in zip(weights,code.support))
    delta=z*target/(np.linalg.norm(z)+1e-12)
    mu=sum(w*D.anchors[n] for w,n in zip(weights,code.support))
    gate=max(0.,float(delta@mu/(np.linalg.norm(delta)*np.linalg.norm(mu)+1e-12)))
    def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)
    gg=tensor(gate)
    ops=[]
    for l in D.layers:
        Q=np.linalg.qr(np.concatenate([D.per_layer[l].bases[n] for n in code.support],axis=1))[0]
        qq,dd,mm=tensor(Q),tensor(D.split(delta)[l]),tensor(D.split(mu)[l])
        def op(h,q=qq,d=dd,m=mm,g=gg):return compiled_kernel(h.contiguous(),q,d,m,g)
        ops.append(op)
    return ops,list(D.layers)
