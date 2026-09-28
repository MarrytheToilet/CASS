"""Equivalent weighted group-LASSO updates using an offline Gram cache.

Maintains A.T @ residual, instead of the hidden-dimensional residual. Uses the
same lambda path, permutation seed, stopping rule, support cap, and weights as
cass.solver. Precomputation belongs to the reusable offline dictionary.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cass.solver import SparseCode


class GramSolver:
    def __init__(self,D):
        self.D = D
        self.names = D.task_names
        self.A = np.concatenate([D.bases[n] for n in self.names],axis=1)
        self.H = self.A.T @ self.A
        self.slices = {}
        offset = 0
        for n in self.names:
            rank = D.bases[n].shape[1]
            self.slices[n] = slice(offset,offset+rank)
            offset += rank
        self.weights = {n: np.sqrt(self.slices[n].stop-self.slices[n].start) for n in self.names}

    def subset(self,names):
        """Reuse cached inner products for a shortlist, preserving its order."""
        names=list(names)
        assert len(names)==len(set(names)) and set(names)<=set(self.names)
        sub=object.__new__(GramSolver)
        sub.D=self.D.subset(names);sub.names=names
        indices=np.concatenate([np.arange(self.slices[n].start,self.slices[n].stop) for n in names])
        sub.A=self.A[:,indices].copy()
        sub.H=self.H[np.ix_(indices,indices)].copy()
        sub.slices={};offset=0
        for n in names:
            rank=self.slices[n].stop-self.slices[n].start
            sub.slices[n]=slice(offset,offset+rank);offset+=rank
        sub.weights={n:self.weights[n] for n in names}
        return sub

    def fit(self,z,lam,warm_start=None,max_iter=200,tol=1e-6,b=None):
        z = np.asarray(z,dtype=np.float64)
        b = self.A.T @ z if b is None else b
        x = (np.concatenate([warm_start[n] for n in self.names]).copy()
             if warm_start else np.zeros(self.A.shape[1]))
        gradient = b-self.H@x
        rng = np.random.default_rng(0)
        for _ in range(max_iter):
            max_rel = 0.
            for n in rng.permutation(self.names):
                sl = self.slices[n]
                old = x[sl].copy()
                block = gradient[sl]+self.H[sl,sl]@old
                norm = np.linalg.norm(block)
                threshold = lam*self.weights[n]
                new = np.zeros_like(block) if norm <= threshold else (1-threshold/norm)*block
                change = new-old
                change_norm = np.linalg.norm(change)
                max_rel = max(max_rel,change_norm/(np.linalg.norm(old)+1e-8))
                if change_norm:
                    gradient -= self.H[:,sl]@change
                    x[sl] = new
            if max_rel < tol:
                break
        coeffs = {n:x[sl].copy() for n,sl in self.slices.items()}
        delta = self.A@x
        support = [n for n in self.names if np.linalg.norm(coeffs[n])>1e-10]
        residual = np.linalg.norm(z-delta)/(np.linalg.norm(z)+1e-12)
        return SparseCode(coeffs,support,delta,residual,lam)

    def solve(self,z_list,s_max=5,n_points=20,floor=.02):
        z = np.mean(np.asarray(z_list,dtype=np.float64),axis=0)
        b = self.A.T@z
        lmax = max(np.linalg.norm(b[self.slices[n]])/self.weights[n] for n in self.names)
        if lmax <= 0.:
            return self.fit(z,0.,b=b)
        best,warm = None,None
        for lam in np.geomspace(.99*lmax,floor*lmax,n_points):
            code = self.fit(z,lam,warm_start=warm,b=b)
            warm = code.coeffs
            if len(code.support)<=s_max and (best is None or code.residual<best.residual):
                best = code
            if len(code.support)>s_max:
                break
        return best if best is not None else self.fit(z,lmax,b=b)
