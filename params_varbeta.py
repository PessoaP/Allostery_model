import numpy as np
from numba import njit
import stsp
import smn

@njit
def triangle(t,beta_T,beta_max):
    return ((t % beta_T)/beta_T)*beta_max
@njit
def const(t,beta_T,beta_max):
    return beta_max
@njit
def step(t,beta_T,beta_max):
    return beta_max*(np.floor(t/beta_T)%2 == 0)

dic = {'triangle': triangle,
       'constant': const,
       'step': step}

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



@njit
def arr_times_A_obeta(arr,cgu):
    #cgu - can go up
    res = -1.0*arr*cgu
    res[1:] += arr[:-1]*cgu[:-1]
    return res

@njit
def arr_times_A(arr,t,beta_f,beta_T,beta_max,cgu,A_fixed):
    beta = beta_f(t,beta_T,beta_max)
    return beta*arr_times_A_obeta(arr,cgu) + smn.array_times_sm(arr,A_fixed)

@njit
def evolve_RK(rho,t,dt,    beta_f,beta_T,beta_max,can_go_up,A_fixed):
    k1 = arr_times_A(rho            , t     , beta_f,beta_T,beta_max,can_go_up,A_fixed)
    k2 = arr_times_A(rho + k1*(dt/2), t+dt/2, beta_f,beta_T,beta_max,can_go_up,A_fixed)
    k3 = arr_times_A(rho + k2*(dt/2), t+dt/2, beta_f,beta_T,beta_max,can_go_up,A_fixed)
    k4 = arr_times_A(rho + k3*dt    , t+dt  , beta_f,beta_T,beta_max,can_go_up,A_fixed)

    return rho + (dt/6)*(k1+k2+k2+k3+k3+k4)
    
@njit
def RK_solve(init,t_init,t_final,dt,    beta_f,beta_T,beta_max,can_go_up,A_fixed):     
    t = t_init*1.0 #float
    rho = init*1.0 #array

    for i in range( int((t_final - t_init) / dt) ):
        rho = evolve_RK(rho,t,dt, beta_f,beta_T,beta_max,can_go_up,A_fixed)
        t+=dt
       
    return evolve_RK(rho,t,t_final-t, beta_f,beta_T,beta_max,can_go_up,A_fixed)

class A_variable:
    def __init__(self,beta_max,beta_T,value_nbeta,can_go_up,N,function='triangle'):
        S_ind = stsp.getS(N)
        rate_matrix_fixed = smn.sparse_matrix(*stsp.get_rate_matrix(np.concatenate([[0],value_nbeta])
                                                                    ,S_ind,N))

        self.A_fixed = rate_matrix_fixed - smn.sparse_matrix(np.arange(rate_matrix_fixed.shape,dtype=int),
                                                             np.arange(rate_matrix_fixed.shape,dtype=int),
                                                             rate_matrix_fixed.line_sum())
        
        self.beta_max = 1.0*beta_max
        self.beta_T = 1.0*beta_T
        self.can_go_up = can_go_up
        self.beta_f = dic[function]
    
    def arr_times_A(self,rho,t):
        return arr_times_A(rho,t,    self.beta_f,self.beta_T,self.beta_max,self.can_go_up,self.A_fixed)


    def evolve_RK(self,rho,t,dt):
        return evolve_RK(rho,t,dt,    self.beta_f,self.beta_T,self.beta_max,self.can_go_up,self.A_fixed)

    
    def solve(self,init,t_init,t_final,dt):
        return RK_solve(init,t_init,t_final,dt,    self.beta_f,self.beta_T,self.beta_max,self.can_go_up,self.A_fixed)

class case:
    def __init__(self,initial,value_nbeta,beta_max,function_params,function='triangle'):

        self.value_nbeta = value_nbeta
        self.beta_max = beta_max
        self.beta_T = function_params
        self.beta_f = lambda t: dic[function](t,
                                              self.beta_T,
                                              self.beta_max)

        gamma_s = value_nbeta[0]
        max_mean = beta_max/gamma_s

        Ns = int(max_mean + 10*np.sqrt(max_mean) + 1)
        N = np.array((4,2,Ns,Ns))
        self.N = N

        self.states = stsp.make_stsp(initial,N)
        self.pinitial = stsp.make_initial(initial,self.states)
        self.can_go_up = self.states[:,-1]!=(Ns-1)

        self.A = A_variable(beta_max,self.beta_T,value_nbeta,self.can_go_up,self.N,function)
        self.dt = .9/(np.abs(self.A.A_fixed.values).max()+beta_max)   

    
    def solver(self,t_init,T_finals,init=None):
        dt = self.dt
        if init is None:
            p = self.pinitial*1.0
        else:
            p = init*1.0
        if isinstance(1.0*T_finals, float) or isinstance(1.0*T_finals, np.float64) or isinstance(1.0*T_finals, np.float32):
            return self.A.solve(p,t_init,1.0*T_finals,dt)
        else:
            t = 1.0*t_init
            pt = []
            for T in T_finals:
                p = self.solver(t,T,p)
                pt.append(p)
                t=T
            return np.vstack(pt)

    #def run_guillespie(self,Ts):
    #    if isinstance(Ts,np.ndarray):
    #        return np.stack([self.run_guillespie(t)[1] for t in Ts])
    #    
    #    return guillespie(self.initial,Ts,self.value)
        

def create_cases(bog,allo_rate=10,function='triangle'):
    init = np.array((0,  #A
                    0,  #B
                    0,  #P
                    bog*1.0 #S
                    ))
    
    allosteric_value = np.array((bog*1.0, #beta_s
                                1.,  #gamma_s
                                1,   #kAon
                                1,   #kAoff
                                10,  #kApon
                                1.,  #kApoff
                                1.,  #alpha
                                4.,  #alphap
                                10., #alpha_s
                                1.,  #alpha_sp
                                1.,  #nu
                                allo_rate*1.0, #nup
                                1.,  #kBon
                                1.,  #kBoff
                                1.   #gammaP
                                ))

    non_allost_value = np.array((bog*1.0, #beta_s
                                1., #gamma_s
                                11, #kAon
                                2,  #kAoff
                                0,  #kApon
                                0,  #kApoff
                                0., #alpha
                                0,  #alphap
                                0., #alpha_s
                                0., #alpha_sp
                                (1.+allo_rate),  #nu
                                0., #nup
                                1., #kBon
                                1., #kBoff
                                1.  #gammaP
                                ))
    
    return case(init,allosteric_value[1:],bog,10.,function), case(init,non_allost_value[1:],bog,10.,function)
