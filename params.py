import numpy as np
from numba import njit
import stsp
import smn

@njit
def get_B(sm):
    newlines=np.arange(sm.shape)
    b_lines=np.concatenate((newlines,sm.lines))
    b_columns=np.concatenate((newlines,sm.columns))

    ad =  smn.sm_sum(sm.lines,sm.columns,sm.values,sm.shape)
    omega = 1.1*ad.max()
    b_values = np.concatenate(((1-ad/omega),(sm.values/omega)))
    return smn.sparse_matrix(b_lines,b_columns,b_values,sm.shape),omega

@njit
def solve(rho_init,B,omegaT):
    rho = rho_init
    log_pf = (-omegaT) #poisson factor log
    pf_cum = np.exp(log_pf)

    ans = rho*np.exp(log_pf)
    lim = omegaT+6*np.sqrt(omegaT)
    for k in np.arange(1,lim):
        log_pf  += np.log(omegaT/k)
        pf = np.exp(log_pf)
        pf_cum += pf

        rho = smn.array_times_sm(rho,B)
        ans += rho*pf

    ans += rho*(1.0-pf_cum)
    return ans

@njit
def guillespie(initial,T,value):
    t = 0.
    x = initial

    while True:
        t_p,x_p = stsp.reactions.Gillespie_evolve(x,t,value)
        if t_p>T:
            return t,x
        else:
            t,x = t_p,x_p

class case:
    def __init__(self,initial,value):
        self.initial = initial
        self.value = value

        beta_s,gamma_s = value[:2]
        mean_steady = beta_s/gamma_s
        Np = int(mean_steady + 6*np.sqrt(mean_steady) + 1)

        Na,Nb = initial[:4].sum(),initial[4:6].sum()
        st = stsp.make_stsp(Na,Nb,Np,Np)
        self.states = st
        self.pinitial = np.zeros(st.shape[0])

        ind = stsp.search(initial,st,st.shape[0]//2)
        self.pinitial[ind]=1.0

        rate_matrix = smn.sparse_matrix(*stsp.get_rate_matrix(value,st))
        self.B,self.omega = get_B(rate_matrix)

    def solver(self,Ts):
        p = self.pinitial*1.0
        pt = []
        t=0
        for T in Ts:
            p = solve(p,self.B,(T-t)*self.omega)
            pt.append(p)
            t=T
        return np.vstack(pt)
    
    def run_guillespie(self,Ts):
        if isinstance(Ts,np.ndarray):
            return np.stack([self.run_guillespie(t)[1] for t in Ts])
        
        return guillespie(self.initial,Ts,self.value)