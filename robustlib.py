import numpy as np
import scipy
from scipy import special
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mpld3
mpld3.enable_notebook()

class RunCollection(object):
    def __init__(self, func, params, bounds):
        self.runs = []
        self.func = func
        self.params = params
        self.bounds = bounds
        
    def run(self, trials):
        for i in range(trials):
            self.runs.append(self.func(self.params, self.bounds))

class DenseNoiseModel(object):
    def __init__(self, dist):
        self.dist = dist

    def generate(self, d, k, eps, m):
        tm = np.append(np.ones(k), np.zeros(d-k))

        G = np.random.randn(m, d) + tm

        S = G.copy()

        L = int(m*(1-eps))
        S[L:] += self.dist

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = (S, indicator, k, eps, 0.2)
        return params, tm

def get_error(f, params, tm):
    return LA.norm(tm - f(params))


class BimodalModel(object):
    def __init__(self):
        pass

    def generate(self, d, k, eps, m):
        tm = np.append(np.ones(k), np.zeros(d-k))
        fm = np.append(np.zeros(d-k), np.ones(k))

        cov = 2*np.identity(d) - np.diag(tm)

        G = np.random.randn(m, d) + tm
        G2 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
        G1 = np.random.randn(m, d) + fm

        S = G.copy()

        L = int(m*(1-eps))
        M = int((L + m)/2)

        S[L:M] = G1[L:M]
        S[M:] = G2[M:]

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = (S, indicator, k, eps, 0.2)
        return params, tm


class Params(object):
    def __init__(self, d, m, eps, k, tau, S, fdr, plotl, plotc, plot):
        self.d = d
        self.m = m
        self.eps = eps
        self.k = k
        self.tau = tau
        self.S = S
        self.indicator = indicator

""" Tail estimates """

def tail_m(T, params):
    eps, k, m, d, tau = params.eps, params.k, params.m, params.d, params.tau    

    return (special.erfc(T/np.sqrt(2)) + (eps**2)/(np.log(k*np.log(m*d/tau))*T**2))

def tail_c(T, params): 
    eps, k, m, d, tau = params.eps, params.k, params.m, params.d, params.tau    

    idx = np.nonzero((T < 6))
    v = 3*np.exp(-T/3) + (eps**2/(T*(np.log(T)**2)))
    v[idx] = 1
    return v

""" P(x) for quadratic filter """

def p(X, mu, M):
    F = LA.norm(M)
    D = X - mu
    vec = (D.dot(M) * D).sum(axis=1)
    return (vec - np.trace(M.T.dot(M)))/F


""" Filters """

def filter_m_sp(params, ev, v, fdr = 0.1, plot = 0, f = 0):
    """This filters elements of S whenever the 
    deviation of <x, v> from the median 
    is more than reasonable. 
    """
    S = params.S
    indicator = params.indicator
    eps = params.eps
    m = params.m
    d = params.d

    l = len(S)
    dots = S.dot(v)
    m2 = np.median(dots)

    x = np.abs(dots - m2) - 3*np.sqrt(eps*ev)
    p_x = tail_m(np.abs(dots - m2), eps, k, m, d, tau)
    p_x[p_x > 1] = 1
    
    sorted_idx = np.argsort(p_x)
    sorted_p = p_x[sorted_idx]

        
    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
    if T > 0.51*m : T = 0

    idx = np.nonzero((p_x >= sorted_p[T]))
    
    if len(S)==len(idx[0]):
        tfdr = 0
    else:
        tfdr = (sum(indicator) - sum(indicator[idx]))/(len(S)-len(idx[0]))
        
    if plot==1:
        plt.plot(np.arange(l), sorted_p)
        plt.plot(T*np.ones(100), 0.01*np.arange(100), linewidth=3)
        plt.plot(np.arange(l), indicator[sorted_idx], linestyle='-.', linewidth=3)
        plt.plot([0,len(S)],[0,fdr], '--')
        plt.title("Linear filter: sample size {}, T = {}, True FDR = {}".format(m, T, tfdr))
        plt.xlabel("Experiments")
        plt.ylabel("p-values")
        plt.figure(f)
        
    return idx

def filter_c_sp(params, M_u, mu, fdr = 0.1, plot = 0, f = 0):
    
    """
    This filters elements of S whenever the 
    degree 2 polynomial p(X) is larger
    than reasonable.
    """

    S = params.S
    indicator = params.indicator
    eps = params.eps
    k = params.k
    tau = params.tau
    
    l = len(S)
    m, d = S.shape
    p_x = tail_c(np.abs(p(S, mu, M_u)), eps, k, m, d, tau)
    x = np.abs(p(S, mu, M_u))

    p_x[p_x > 1] = 1
    sorted_idx = np.argsort(p_x)
    sorted_p = p_x[sorted_idx]

    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0.01)[::-1])
    print((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
    print((fdr/l)*np.arange(l))
    if T==l: T = 0
    print("T = ", T)
    
    idx = np.nonzero((p_x >= sorted_p[T] ))
    if len(S)==len(idx[0]):
        tfdr = 0
    else:
        tfdr = (sum(indicator) - sum(indicator[idx]))/(len(S)-len(idx[0]))

    if plot==1:
        plt.plot(np.arange(l), sorted_p)
        plt.plot(T*np.ones(100), 0.01*np.arange(100), linewidth=3)
        plt.plot(np.arange(l), indicator[sorted_idx], linestyle='-.',linewidth=3)
        plt.plot([0,len(S)],[0,fdr], '--')
        plt.title("Quadratic Filter: sample size {}, T = {}, True FDR = {}".format(m, T, tfdr))
        plt.xlabel("Experiments")
        plt.ylabel("p-values")
        plt.figure(f)

    return idx

""" Auxillary functions """

""" Naive Prune """

def NP(params):
    
    k = params.k
    eps = params.eps
    tau = params.tau
    m = params.m
    d = params.d
    
    T_naive = np.sqrt(2*np.log(m*d/tau))
   
    med = np.median(params.S, axis=0)
    idx = (np.max(np.abs(med-params.S), axis=1) < T_naive)
    
    return(idx)

""" Threshold to top-k in absolute value """

def topk_abs(v, k):
    u = np.argpartition(np.abs(v), -k)[-k:]
    z = np.zeros(len(v))
    z[u] = v[u]
    return z

""" RME algorithms """ 

def NPmean_sp(params):
    k = params.k
    S = params.S
    
    S = S[NP(params)]
    mean = np.mean(S, axis=0)

    if m > len(S): print("NP_sp pruned!")

    return topk_abs(mean, k) 

def RME_sp(params, plotl = 0, plotc = 0, f = 0, fdr=0.2):
    k = params.k
    d = params.d
    m = params.m
    eps = params.eps
    tau = params.tau
    
    idx = NP(params)
    
    params.S = params.S[idx]
    params.indicator = params.indicator[idx]

    mu_e = np.mean(params.S, axis=0) 

    while True:
        
        if len(params.S)==0: 
            print("No points remaining.")
            return topk_abs(mu_e)

        if len(params.S)==1: 
            print("1 point remaining.")
            return topk_abs(mu_e)
        
        mu_e = np.mean(params.S, axis=0) 
        cov_e = np.cov(params.S, rowvar=0)
        
        M = cov_e - np.identity(d) 
        (mask, u) = indicat(M, k)
        M_mask = mask*M

        if LA.norm(M_mask) < eps*np.log(1/eps): 
            print("Valid output")
            return topk_abs(mu_e)
        
        cov_u = cov_e[np.ix_(u,u)]
        ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1,k-1))
        v = v.reshape(len(v),)
                
        if ev > 1 + eps*np.sqrt(np.log(1/eps)): 

            print("Linear filter.")
            x = len(params.S)
            params.S = params.S[np.ix_(np.arange(len(S)),u)]
            params.m = len(params.S)
            
            idx = filter_m_sp(params, ev, v, fdr = fdr, plot=plotl, f=f)
            f+=1
            params.S =  params.S[idx]
            params.indicator = params.indicator[idx]

            if x == params.m: 
                print("Linear filter did not filter anything.")

        mu_e = np.mean(params.S, axis=0)

        print("Quadratic filter.")
        x = len(params.S)
        idx = filter_c_sp(params, M_mask, mu_e, fdr = fdr, plot=plotc, f=f)
        f+=1

        params.S =  params.S[idx]
        params.indicator = params.indicator[idx]
        params.m = len(S)

        if x == params.m: 
            print("Quadratic filter did not filter anything.")
            return topk_abs(mu_e)
        
