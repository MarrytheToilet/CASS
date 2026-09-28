"""Independent sum-of-projectors reference and contraction checks on CPU."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from common import HERE
from projector_refine import WeightedFactory


def main():
    rng=np.random.default_rng(20260927);torch.set_num_threads(1)
    errors=[];ratios=[]
    for draw in range(60):
        d=48;layers=[12,16];names=[str(i) for i in range(1+draw%5)]
        weights=rng.random(len(names));weights/=weights.sum()
        blocks={l:{n:np.linalg.qr(rng.normal(size=(d,2+int(n)%5)))[0] for n in names} for l in layers}
        clean={n:{l:rng.normal(size=d) for l in layers} for n in names}
        delta=rng.normal(size=2*d);gate=float(rng.random());strength=float(rng.random());gamma=1.5
        D=SimpleNamespace(d=d,layers=layers,per_layer={l:SimpleNamespace(bases=blocks[l]) for l in layers},
                          split=lambda x:{l:x[i*d:(i+1)*d] for i,l in enumerate(layers)})
        base=SimpleNamespace(D=D,code=SimpleNamespace(support=names),w=weights,clean=clean,delta=delta,gate=gate)
        factory=WeightedFactory(base,device='cpu')
        h=rng.normal(size=(7,d)).astype('float32');other=rng.normal(size=(7,d)).astype('float32')
        for mode in ['weighted','rank8','rank16','rank32']:
            ops,_=factory.ops(mode,gamma,strength)
            for l,op in zip(layers,ops):
                output=op(torch.from_numpy(h)).numpy();second=op(torch.from_numpy(other)).numpy()
                ratio=float(np.linalg.norm(output-second)/np.linalg.norm(h-other));ratios.append(ratio)
                assert ratio<=1+1e-6
                if mode=='weighted':
                    shifted=h.astype('float64')+gamma*D.split(delta)[l]
                    correction=sum(w*((clean[n][l]-shifted)@blocks[l][n])@blocks[l][n].T for w,n in zip(weights,names))
                    reference=shifted+strength*gate*correction
                    error=float(np.max(np.abs(output-reference)));errors.append(error)
                    np.testing.assert_allclose(output,reference,atol=2e-6,rtol=2e-6)
                    M=sum(w*blocks[l][n]@blocks[l][n].T for w,n in zip(weights,names))
                    eigen=np.linalg.eigvalsh(M);assert eigen.min()>-1e-12 and eigen.max()<1+1e-12
    result=dict(n_fixed_map_checks=len(ratios),n_explicit_weighted_sum_checks=len(errors),
                maximum_absolute_float32_error=max(errors),maximum_contraction_ratio=max(ratios),
                scope='CPU functional-equivalence and fixed-state geometry checks; no generation-quality claim.')
    (HERE/'projector_geometry_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
