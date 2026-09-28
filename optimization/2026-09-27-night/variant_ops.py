"""Exploratory context anchors and weighted low-rank support geometry."""
import numpy as np
import torch


class OperatorFactory:
    def __init__(self,D,code,zlist,clean_means):
        self.D=D;self.code=code
        z=np.mean(zlist,axis=0)
        w=np.array([np.linalg.norm(code.coeffs[n]) for n in code.support]);w=w/w.sum() if len(w) else w
        target=sum(a*np.linalg.norm(D.anchors[n]) for a,n in zip(w,code.support))
        self.delta=z*target/(np.linalg.norm(z)+1e-12) if len(w) else z
        mu=sum((a*D.anchors[n] for a,n in zip(w,code.support)),start=np.zeros_like(z))
        self.gate=max(0.,float(z@mu/(np.linalg.norm(z)*np.linalg.norm(mu)+1e-12)))
        self.w=w;self.mu=mu;self.clean=clean_means;self.cache={}

    def geometry(self,kind):
        if kind in self.cache:return self.cache[kind]
        D=self.D;support=self.code.support;w=self.w
        def tensor(a):return torch.as_tensor(a,device='cuda',dtype=torch.float32)
        result=[]
        for l in D.layers:
            if support:
                B=np.concatenate([D.per_layer[l].bases[n] for n in support],axis=1)
                if kind.startswith('rank'):
                    rank=int(kind[4:])
                    BW=np.concatenate([np.sqrt(a)*D.per_layer[l].bases[n] for a,n in zip(w,support)],axis=1)
                    Q=np.linalg.svd(BW,full_matrices=False)[0][:,:rank]
                else:Q=np.linalg.qr(B)[0]
                anchor=(sum(a*self.clean[n][l] for a,n in zip(w,support)) if kind.startswith('context')
                        else D.split(self.mu)[l])
            else:Q=np.zeros((D.d,0));anchor=np.zeros(D.d)
            result.append((tensor(Q),tensor(D.split(self.delta)[l]),tensor(anchor)))
        self.cache[kind]=result
        return result

    def ops(self,kind='original',gamma=1.,correction=1.):
        ops=[];gate=self.gate
        for Q,dv,mu in self.geometry(kind):
            def op(h,q=Q,d=dv,m=mu):
                h=h.float();diff=m-h;proj=(diff@q)@q.T
                if kind.startswith('context'):
                    delta=d-(correction*gate*((d@q)@q.T) if kind=='context_residual' else 0.)
                    return h+gamma*delta+correction*gate*proj
                alpha=(2*(diff-proj).norm(dim=1)/(h.norm(dim=1)+1e-8)).clamp(max=1).unsqueeze(1)
                return h+(gate*alpha+1-gate)*gamma*d+correction*gate*alpha*proj
            ops.append(op)
        return ops,list(self.D.layers)
