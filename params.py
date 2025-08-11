import numpy as np
from numba import njit
import stsp
import smn

@njit
def get_B(sm):
    newrows=np.arange(sm.shape)
    b_rows=np.concatenate((newrows,sm.rows))
    b_columns=np.concatenate((newrows,sm.columns))

    ad =  smn.sm_sum(sm.rows,sm.columns,sm.values,sm.shape)
    omega = 1.1*ad.max()
    b_values = np.concatenate(((1-ad/omega),(sm.values/omega)))
    return smn.sparse_matrix(b_rows,b_columns,b_values,sm.shape),omega

@njit
def solve(rho_init,B,omegaT):
    if omegaT<1e-8:
        return rho_init
    rho = rho_init
    log_pf = (-omegaT) #poisson factor log
    pf_cum = np.exp(log_pf)

    ans = rho*np.exp(log_pf)
    lim = max(omegaT+6*np.sqrt(omegaT),5)
    for k in np.arange(1,lim):
        log_pf  += np.log(omegaT/k)
        pf = np.exp(log_pf)
        pf_cum += pf

        rho = smn.array_times_sm(rho,B)
        ans += rho*pf

    ans += rho*(1.0-pf_cum)
    return ans

@njit
def gillespie(initial,T,value):
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
        self.value = value

        beta_s,gamma_s = value[:2]
        mean_steady = beta_s/gamma_s

        Ns = max(11,int(mean_steady + 10*np.sqrt(mean_steady) + 1))
        N = np.array((4,2,Ns,Ns))
        self.N = N

        self.states = stsp.make_stsp(initial,N)
        self.pinitial = stsp.make_initial(initial,self.states)

        S_ind = stsp.getS(N)

        sm = stsp.get_rate_matrix(value,S_ind,N)
        rate_matrix = smn.sparse_matrix(*sm)
        self.B,self.omega = get_B(rate_matrix)
   
    def solver(self,Ts,init=None):
        if init is None:
            p = self.pinitial*1.0
        else:
            p = init*1.0
        if isinstance(1.0*Ts, float):
            return solve(p,self.B,(Ts)*self.omega)
        else:
            t = 0.
            pt = []
            for T in Ts:
                p = self.solver(T-t,init=p)
                pt.append(p)
                t=T
            return np.vstack(pt)

    def run_gillespie(self,Ts):
        if isinstance(Ts,np.ndarray):
            return np.stack([self.run_gillespie(t)[1] for t in Ts])
        
        return gillespie(self.initial,Ts,self.value)
    
    def found_steady_state(self,p,tol=1e-12):
        pA_overomega = (smn.array_times_sm(p,self.B)-p)
        return np.max(np.abs(pA_overomega))*self.omega<tol
    
    def find_steady(self):
        p=self.pinitial
        is_steady=False
        t=0
        while not(is_steady):
            is_steady = self.found_steady_state(p)
            p = self.solver(10,p)
            t+=10
        return p,t
    
def V_create_cases(bog,V_allo_rate=10):
    init = np.array((0,  #A
                    0,  #B
                    0,  #P
                    bog*1.0 #S
                    ))
    
    allosteric_value = np.array((bog*1.0, #beta_s
                                1.,   #gamma_s
                                1,  #kAon
                                1,  #kAoff
                                10,  #kApon
                                1.,  #kApoff
                                1.,  #alpha
                                4.,  #alphap
                                10.,  #alpha_s
                                1.,  #alpha_sp
                                1.,  #nu
                                V_allo_rate*1.0, #nup
                                1., #kBon
                                1., #kBoff
                                1.  #gammaP
                                ))

    non_allost_value = np.array((bog*1.0, #beta_s
                                1.,   #gamma_s
                                11,  #kAon
                                2,  #kAoff
                                0,  #kApon
                                0,  #kApoff
                                0.,  #alpha
                                0,  #alphap
                                0.,  #alpha_s
                                0.,  #alpha_sp
                                (1.+V_allo_rate),  #nu
                                0., #nup
                                1., #kBon
                                1., #kBoff
                                1.  #gammaP
                                ))
    
    return case(init,allosteric_value), case(init,non_allost_value)

def K_create_cases(bog,K_allo_rate=10):
    init = np.array((0,  #A
                    0,  #B
                    0,  #P
                    bog*1.0 #S
                    ))
    
    allosteric_value = np.array((bog*1.0, #beta_s
                                1.,   #gamma_s
                                1,  #kAon
                                1,  #kAoff
                                K_allo_rate*1.0,  #kApon
                                1.,  #kApoff
                                1.,  #alpha
                                4.,  #alphap
                                10.,  #alpha_s
                                1.,  #alpha_sp
                                1.,  #nu
                                10, #nup
                                1., #kBon
                                1., #kBoff
                                1.  #gammaP
                                ))

    non_allost_value = np.array((bog*1.0, #beta_s
                                1.,   #gamma_s
                                (1+K_allo_rate),  #kAon
                                2,  #kAoff
                                0,  #kApon
                                0,  #kApoff
                                0.,  #alpha
                                0,  #alphap
                                0.,  #alpha_s
                                0.,  #alpha_sp
                                11,  #nu
                                0., #nup
                                1., #kBon
                                1., #kBoff
                                1.  #gammaP
                                ))
    
    return case(init,allosteric_value), case(init,non_allost_value)