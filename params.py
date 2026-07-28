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
    lim = max(omegaT+6*np.sqrt(omegaT),0) +5
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
        print(Ns)
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
    
    def found_steady_state(self,p,tol=1e-6):
        pA_overomega = (smn.array_times_sm(p,self.B)-p)
        return np.max(np.abs(pA_overomega))*self.omega<tol
    
    def find_steady(self):
        p=self.pinitial
        is_steady=False
        t=0
        while not(is_steady):
            is_steady = self.found_steady_state(p)
            p = self.solver(5,p)
            t+=5
        return p,t


def get_init(bog):
    return np.array((0,  # A
                     0,  # B
                     0,  # P
                     bog*1.0  # S
                     ), dtype=float)

@njit
def allosteric_rates(base_alpha,base_alphap,
                     base_kon,base_koff,
                     K_allo_rate, variant = 'C2'):
    alpha = base_alpha
    alphap = base_alphap

    alphaS  = base_alpha #*K_allo_rate
    alphaSp = base_alphap

    kon = base_kon
    koff = base_koff

    kApon  = base_kon #*sqrtK
    kApoff = base_koff #/sqrtK

    print('variant =', variant)
    if variant == 'C1':
        sqrtK = np.sqrt(K_allo_rate)

        alphaS  = base_alpha*K_allo_rate
        alphaSp = base_alphap

        kApon  = base_kon*K_allo_rate
        kApoff = base_koff

    elif variant == 'C2':
        sqrtK = np.sqrt(K_allo_rate)

        alphaS  = base_alpha*K_allo_rate
        alphaSp = base_alphap

        kApon  = base_kon*sqrtK
        kApoff = base_koff/sqrtK

    elif variant == 'C3':
        sqrtK = np.sqrt(K_allo_rate)

        alphaS  = base_alpha*sqrtK
        alphaSp = base_alphap/sqrtK

        kApon  = base_kon*K_allo_rate
        kApoff = base_koff/sqrtK

    elif variant == 'C4':
        sqrtK = np.sqrt(K_allo_rate)

        alphaS  = base_alpha*sqrtK
        alphaSp = base_alphap/sqrtK

        kApon  = base_kon*sqrtK
        kApoff = base_koff/sqrtK

    elif variant == 'C5':
        sqrtK = np.sqrt(K_allo_rate)

        alpha  = base_alpha/sqrtK
        alphap = base_alphap*sqrtK

        kon  = base_kon/sqrtK
        koff = base_koff*sqrtK

    elif variant == 'C6':
        sqrtK = np.sqrt(K_allo_rate)
        qrtK = np.sqrt(sqrtK)

        alpha  = base_alpha/sqrtK
        alphap = base_alphap*sqrtK

        kon  = base_kon/qrtK
        koff = base_koff*qrtK

        kApon = base_kon*qrtK
        kApoff = base_koff/qrtK 

    elif variant == 'C7':
        sqrtK = np.sqrt(K_allo_rate)
        #qrtK = np.sqrt(sqrtK)

        alpha  = base_alpha/sqrtK
        alphap = base_alphap*sqrtK

        kApon = base_kon*sqrtK
        kApoff = base_koff/sqrtK 

    elif variant == 'C8':
        sqrtK = np.sqrt(K_allo_rate)
        #qrtK = np.sqrt(sqrtK)

        alpha  = base_alpha/K_allo_rate

        kApon = base_kon*sqrtK
        kApoff = base_koff/sqrtK 

    return alpha, alphap, alphaS, alphaSp, kon, koff, kApon, kApoff



def get_allosteric_value(bog,V_allo_rate=1,K_allo_rate=1,
                         variant='C2',
                         base_nu=1,base_kon=1,
                         base_koff=1,base_alpha=1/2,base_alphap=1/4):

    alpha, alphap, alphaS, alphaSp, kon, koff, kApon, kApoff = allosteric_rates(base_alpha,base_alphap,
                                                                                base_kon,base_koff,
                                                                                K_allo_rate,variant=variant)

    return np.array((bog*1.0,  # beta_s
                     1.,       # gamma_s
                     kon,      # kAon
                     koff,     # kAoff
                     kApon,    # kApon
                     kApoff,   # kApoff
                     alpha,    # alpha
                     alphap,   # alphap
                     alphaS,   # alpha_s
                     alphaSp,  # alpha_sp
                     base_nu,  # nu
                     V_allo_rate*base_nu, # nup
                     base_kon, # kBon
                     base_koff,# kBoff
                     1.        # gammaP
                     ), dtype=float)


def get_equivalent_non_allo_value(bog,eqV_allo_rate=1,eqK_allo_rate=1,
                                  variant='C2',
                                  base_nu=1,base_kon=1,
                                  base_koff=1,alpha_base=1/4,alphap_base=1/2):


    #alphaS, alphaSp, kApon, kApoff = allosteric_rates(alpha,alphap,base_kon,base_koff,eqK_allo_rate)

    alpha, alphap, alphaS, alphaSp, kon, koff, kApon, kApoff = allosteric_rates(alpha_base,alphap_base,
                                                                                base_kon,base_koff,
                                                                                eqK_allo_rate,variant=variant)

    denom = alpha + alphap
    kAon_eff  = (alphap*kon  + alpha*kApon)  / denom
    kAoff_eff = (alphap*koff + alpha*kApoff) / denom

    nu_eff = base_nu*(alphaSp + alphaS*eqV_allo_rate)/(alphaS + alphaSp)

    return np.array((bog*1.0,  # beta_s
                     1.,       # gamma_s
                     kAon_eff, # kAon
                     kAoff_eff,# kAoff
                     0.,       # kApon
                     0.,       # kApoff
                     0.,       # alpha
                     0.,       # alphap
                     0.,       # alpha_s
                     0.,       # alpha_sp
                     nu_eff,   # nus
                     0.,       # nup
                     base_kon, # kBon
                     base_koff,# kBoff
                     1.        # gammaP
                     ), dtype=float)


def create_cases(bog, V_allo_rate=1, K_allo_rate=1,
                 base_nu=1, base_kon=1,
                 variant='C2'):
    init = get_init(bog)
    val  = get_allosteric_value(bog, V_allo_rate, K_allo_rate,
                                variant=variant,
                                base_nu=base_nu, base_kon=base_kon)
    return case(init, val)


def create_equivalent_non_allo(bog, eqV_allo_rate=1, eqK_allo_rate=1,
                               variant='C2',
                               base_nu=1, base_kon=1):
    init = get_init(bog)
    val  = get_equivalent_non_allo_value(bog, eqV_allo_rate, eqK_allo_rate,
                                         variant=variant,
                                         base_nu=base_nu, base_kon=base_kon)
    return case(init, val)