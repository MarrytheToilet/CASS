"""Incremental dictionary growth with the initial shared direction frozen."""
import numpy as np
from cass.dictionary import SkillDictionary,MultiLayerDictionary,rank_by_energy


def build_fixed_shared(G,layers,reference):
    dictionaries={}
    for l in layers:
        initial=reference.per_layer[l];D=SkillDictionary(task_names=list(G[l]),U0=initial.U0.copy())
        for name,activations in G[l].items():
            if name in initial.task_names:
                D.bases[name]=initial.bases[name];D.anchors[name]=initial.anchors[name]
                D.spectra[name]=initial.spectra[name];D.raw_means[name]=initial.raw_means[name]
            else:
                raw=np.asarray(activations,dtype=np.float64).T
                residual=raw-D.U0@(D.U0.T@raw)
                U,s,_=np.linalg.svd(residual,full_matrices=False);r=rank_by_energy(s,.9,16)
                D.bases[name]=U[:,:r];D.anchors[name]=residual.mean(1);D.spectra[name]=s;D.raw_means[name]=raw.mean(1)
        dictionaries[l]=D
    return MultiLayerDictionary(dictionaries)
