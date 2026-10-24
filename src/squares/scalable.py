"""Memory-bounded basin proposals with an analytic penalty gradient.

Large puzzles use pair screening plus direct gradient accumulation, never a
dense constraint Jacobian. Feasibility is restored by center dilation and then
checked independently. This is a heuristic, not an optimality certificate.
"""
import math
import time
from .baseline import grid
from .basin import BudgetExpired,metropolis,propose
from .geometry import Square,validate
from .optimize import _libraries


class PenaltyModel:
    def __init__(self,n,clearance=1e-7,deadline=None):
        self.n=n
        self.clearance=clearance
        self.deadline=deadline
        self.np,_=_libraries()
        self.i,self.j=self.np.triu_indices(n,1)
        self.calls=0

    def evaluate(self,z,weight=400):
        if self.deadline is not None and time.perf_counter()>=self.deadline:
            raise BudgetExpired()
        self.calls+=1
        np,n=self.np,self.n
        a=np.asarray(z[:-1]).reshape(n,3)
        x,y,t=a.T
        c,s=np.cos(t),np.sin(t)
        h=(np.abs(c)+np.abs(s))/2
        dh=(-np.sign(c)*s+np.sign(s)*c)/2
        walls=np.stack((x-h,y-h,z[-1]-x-h,z[-1]-y-h))
        v=np.minimum(walls-self.clearance,0)
        g=np.zeros((n,3))
        g[:,0]=2*weight*(v[0]-v[2])
        g[:,1]=2*weight*(v[1]-v[3])
        g[:,2]=-2*weight*dh*np.sum(v,axis=0)
        gl=1+2*weight*np.sum(v[2:])
        loss=float(z[-1]+weight*np.sum(v*v))
        # Far centers cannot have intersecting unit squares, regardless of angle.
        i,j=self.i,self.j
        dx,dy=x[j]-x[i],y[j]-y[i]
        near=dx*dx+dy*dy<(math.sqrt(2)+2*self.clearance)**2
        i,j,dx,dy=i[near],j[near],dx[near],dy[near]
        delta=t[i]-t[j]
        cd,sd=np.cos(delta),np.sin(delta)
        radius=(1+np.abs(cd)+np.abs(sd))/2
        dr=(-np.sign(cd)*sd+np.sign(sd)*cd)/2
        angles=np.stack((t[i],t[i]+math.pi/2,t[j],t[j]+math.pi/2),axis=1)
        ca,sa=np.cos(angles),np.sin(angles)
        dot=dx[:,None]*ca+dy[:,None]*sa
        gaps=np.abs(dot)-radius[:,None]
        k=np.argmax(gaps,axis=1)
        q=np.arange(len(i))
        v=np.minimum(gaps[q,k]-self.clearance,0)
        co,si=ca[q,k],sa[q,k]
        sign=np.sign(dot[q,k])
        rotation=sign*(-dx*si+dy*co)
        coefficient=2*weight*v
        for axis,component in ((0,co),(1,si)):
            np.add.at(g[:,axis],i,-coefficient*sign*component)
            np.add.at(g[:,axis],j,coefficient*sign*component)
        np.add.at(g[:,2],i,coefficient*(-dr+np.where(k<2,rotation,0)))
        np.add.at(g[:,2],j,coefficient*(dr+np.where(k>=2,rotation,0)))
        loss+=float(weight*(v@v))
        return loss,np.append(g.ravel(),gl)


def restore_feasibility(pose,clearance=1e-7,max_scale=1.20):
    """Find one uniform center dilation sufficient to separate every pair."""
    np,_=_libraries()
    a=np.array(pose,dtype=float,copy=True)
    if not np.all(np.isfinite(a)):
        return None
    n=len(a)
    x,y,t=a.T
    i,j=np.triu_indices(n,1)
    dx,dy=x[j]-x[i],y[j]-y[i]
    near=dx*dx+dy*dy<(math.sqrt(2)+2*clearance)**2
    i,j,dx,dy=i[near],j[near],dx[near],dy[near]
    d=t[i]-t[j]
    radius=(1+np.abs(np.cos(d))+np.abs(np.sin(d)))/2
    angles=np.stack((t[i],t[i]+math.pi/2,t[j],t[j]+math.pi/2),axis=1)
    projected=np.abs(dx[:,None]*np.cos(angles)+dy[:,None]*np.sin(angles))
    distance=np.max(projected,axis=1)
    required=np.divide(radius+clearance,distance,
                       out=np.full_like(radius,np.inf),where=distance>0)
    scale=max(1.0,float(np.max(required,initial=1.0)))*(1+clearance)
    if not math.isfinite(scale) or scale>max_scale:
        return None
    a[:,:2]*=scale
    squares=[Square(float(x),float(y),float(t%(math.pi/2))) for x,y,t in a]
    try:
        side=validate(squares,n)
    except ValueError:
        return None
    return squares,side


def solve_scalable(n,seconds=3.0,seed=0,initial=None,maxiter=18,hops=1000):
    np,minimize=_libraries()
    baseline=grid(n)
    best=list(initial) if initial is not None else baseline
    best_side=validate(best,n)
    if validate(baseline,n)<best_side:
        best,best_side=baseline,validate(baseline,n)
    if not math.isfinite(seconds) or seconds<=0 or maxiter<1:
        raise ValueError("invalid scalable search budget")
    started=time.perf_counter()
    model=PenaltyModel(n,deadline=started+seconds)
    rng=np.random.default_rng(seed)
    current=best
    current_energy=best_side
    upper=max(validate(baseline,n),best_side)+0.25
    history=[]
    status="hop-limit"
    if abs(best_side-math.sqrt(n))<1e-12:
        status="area-bound"
    else:
        try:
            for hop in range(hops):
                pose,move=propose(current,rng,hop,upper)
                z=np.append(pose.ravel(),upper)
                result=minimize(lambda v:model.evaluate(v,400),z,jac=True,
                                method="L-BFGS-B",
                                bounds=[(0,upper),(0,upper),(-math.pi,math.pi)]*n+
                                       [(math.sqrt(n),upper)],
                                options={"maxiter":maxiter,"ftol":1e-10})
                candidate=restore_feasibility(result.x[:-1].reshape(n,3))
                if candidate is not None and candidate[1]<best_side:
                    best,best_side=candidate
                energy=float(result.fun)
                accepted=metropolis(current_energy,energy,0.04,float(rng.random()))
                if accepted:
                    current=[Square(float(x),float(y),float(t))
                             for x,y,t in result.x[:-1].reshape(n,3)]
                    current_energy=energy
                history.append({"hop":hop,"move":move,"accepted":accepted,
                                "candidate_valid":candidate is not None,
                                "best_L":best_side,"seconds":time.perf_counter()-started})
        except BudgetExpired:
            status="time-budget"
    return best,{"n":n,"algorithm":"scalable-penalty-basin","seed":seed,
                 "best_L":best_side,"seconds":time.perf_counter()-started,
                 "status":status,"completed_hops":len(history),
                 "history":history,"evaluations":model.calls}
