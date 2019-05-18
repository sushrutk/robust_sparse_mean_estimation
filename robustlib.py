import numpy as np
import scipy
import copy
from scipy import special
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mpld3
mpld3.enable_notebook()

class RunCollection(object):
    def __init__(self, func, params, bounds = 0, keys = []):
        self.runs = []
        self.func = func
        self.params = params
        self.bounds = bounds
        self.keys = keys
        
    def run(self, trials):
        for i in range(trials):
            self.runs.append(self.func(self.params, self.keys, self.bounds))

class DenseNoiseModel(object):
    def __init__(self, dist):
        self.dist = dist

    def generate(self, d, k, eps, m, tau=0.2):
        tm = np.append(np.ones(k), np.zeros(d-k))

        G = np.random.randn(m, d) + tm

        S = G.copy()

        L = int(m*(1-eps))
        S[L:] += self.dist

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau,S,indicator)
        return params, tm

def get_error(f, params, tm):
    return LA.norm(tm - f(params))


class BimodalModel(object):
    def __init__(self):
        pass
    
    def generate(self, d, k, eps, m, tau=0.2):
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
        params = Params(d,m,eps,k,tau,S,indicator)
        return params, tm


class Params(object):
    def __init__(self, d, m, eps, k, tau, S, indicator):
        self.d = d
        self.m = m
        self.eps = eps
        self.k = k
        self.tau = tau
        self.S = S
        self.indicator = indicator
        

def sparse_samp_loss(model_params, keys, m_bounds):
    (Low, Up, step) = m_bounds
    
    results = {}

    for m in np.arange(Low, Up, step):
        d, k, eps = model_params
        model = BimodalModel()
        params, tm = model.generate(d, k, eps, m)
        O = LA.norm(tm - np.mean(params.S * params.indicator, axis=0))
        
        for f in keys:
            results.setdefault(f.__name__, []).append(LA.norm(tm - f(params))/eps)
            
        results.setdefault('oracle', []).append(O/eps)
        results.setdefault('eps', []).append(1)
    
    return results




#             params = (S.copy(), indicator, k, eps, 0.2)
#             if f == RME:
#                 params = (G.copy(), S.copy(), indicator, m, d, eps, 0.2)
#             if f == RME:
#                 mu_o = f(params)
#                 if mu_o.__class__.__name__ != 'int':
#                     u2 = np.argpartition(mu_o, -k)[-k:]
#                     z = np.zeros(len(mu_o))
#                     z[u2] = mu_o[u2]
#                     results.setdefault(f.__name__, []).append(LA.norm(tm - z)/eps)
#                 else:
#                     results.setdefault(f.__name__, []).append(LA.norm(tm - f(params))/eps)
                    
#             else: 


        
        
#         m = samp
        
#         tm = np.append(np.ones(k), np.zeros(d-k))
#         fm = np.append(np.zeros(d-k), np.ones(k))

#         cov = 2*np.identity(d) - np.diag(tm)
# #         tm = tm/LA.norm(tm)

#         G = np.random.randn(m, d) + tm
#         G2 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
# #         G1 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
#         G1 = np.random.randn(m, d) + fm

#         S = G.copy()

#         indicator = np.zeros(len(S))
#         L = int(m*(1-eps))
#         M = int((L + m)/2)
#         print(m-L)

#         S[L:M] = G1[L:M]
#         S[M:] = G2[M:]

#         indicator = np.ones(len(S))
#         indicator[int(m*(1-eps)):] = 0

#         true_mean = np.append(np.ones(k),np.zeros(d-k))
#         false_mean = np.append(np.zeros(d-k), np.ones(k))
# #         false_mean = np.ones(d)

#         G = np.random.randn(m, d) + true_mean

#         S = G.copy()
# #         S[int(m*(1-eps)):] += dist*false_mean + true_mean
#         S[int(m*(1-eps)):] = true_mean + dist
#         indicator = np.ones(len(S))
#         indicator[int(m*(1-eps)):] = 0
#         params = (S.copy(), indicator, k, eps, 0.2)


        # for func in [RME_sp, NP_sp]:
        # ... func(params)
        # Option 1: set RME_sp.name = 'RME_sp' somewhere, use func.name
        # Option 2: func.__name__

#         for f in [rl.RME_sp_L, rl.NP_sp]:
#             results.setdefault(f.__name__, []).append(LA.norm(true_mean - f(params))/eps)


def plot_l_samples(Run, keys):
    cols = {'RME_sp':'b', 'RME_sp_L':'g', 'RME':'r','ransacGaussianMean':'y' , 'NP_sp':'g'}
    s = len(Run.runs)
    for key in keys:
        A = np.array([res[key] for res in Run.runs])
        xs = np.arange(*Run.bounds)
        plt.plot(xs, np.array(np.median(A,axis = 0)), label=key, color = cols[key])
        mins = [np.sort(x)[int(s*0.25)] for x in A.T]
        maxs = [np.sort(x)[int(s*0.75)] for x in A.T]
        plt.fill_between(xs, mins, maxs, color=cols[key], alpha=0.2)
    d, k, di, eps = Run.params
    plt.title(f'd = {d}, k = {k}, eps = {eps}')
    plt.xlabel('m')
    plt.ylabel('MSE/eps')
    plt.legend()

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
    p_x = tail_m(np.abs(dots - m2), params)
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
    p_x = tail_c(np.abs(p(S, mu, M_u)), params)
    x = np.abs(p(S, mu, M_u))

    p_x[p_x > 1] = 1
    sorted_idx = np.argsort(p_x)
    sorted_p = p_x[sorted_idx]

    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0.01)[::-1])
#     print((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
#     print((fdr/l)*np.arange(l))
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



def indicat(M, k): 
    
    """
    creates an indicator matrix for 
    the largest k diagonal entries and 
    largest k**2 - k off-diagonal entries
    """
    
    ans = np.zeros(M.shape)

    u = np.argpartition(M.diagonal(), -k)[-k:] # Finds largest k indices of the diagonal 
    ans[(u,u)] = 1

    idx = np.where(~np.eye(M.shape[0],dtype=bool)) # Change this too
    val = np.partition(M[idx].flatten(), -k**2+k)[-k**2+k] # (k**2 - k)th largest off-diagonl element
    idx2 = np.where(M > val)
    
    ans[idx2] = 1
    
    return (ans, u)


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

def RME_sp(params, plotl = 0, plotc = 0, f = 0, fdr=0.2, verbose=False):
    k = params.k
    d = params.d
    m = params.m
    eps = params.eps
    tau = params.tau
    S = params.S
    indicator = params.indicator
    
    idx = NP(params)
    
    S = S[idx]
    indicator = indicator[idx]

    mu_e = np.mean(S, axis=0) 

    while True:
        
        if len(S)==0: 
            print("No points remaining.")
            return topk_abs(mu_e, k)

        if len(S)==1: 
            print("1 point remaining.")
            return topk_abs(mu_e, k)
                
        mu_e = np.mean(params.S, axis=0)
        cov_e = np.cov(params.S, rowvar=0)
        
        M = cov_e - np.identity(d) 
        (mask, u) = indicat(M, k)
        M_mask = mask*M

        if LA.norm(M_mask) < eps*np.log(1/eps): 
            print("Valid output")
            return topk_abs(mu_e, k)
        
        cov_u = cov_e[np.ix_(u,u)]
        ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1,k-1))
        v = v.reshape(len(v),)
                
        if ev > 1 + eps*np.sqrt(np.log(1/eps)): 
            if verbose:
                print("Linear filter...")

            x = len(S)
            p2 = copy.copy(params)
            p2.S = S[np.ix_(np.arange(x),u)]
            p2.indicator = indicator
            
            idx = filter_m_sp(p2, ev, v, fdr = fdr, plot=plotl, f=f)
            
            if verbose:
                bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
#                 print(f"Filtered out {x - len(idx[0])}/{x}, {bad_filtered} false ({bad_filtered / (x - len(idx[0])):0.2f} vs {fdr})")
            f+=1
            S =  S[idx]
            params.indicator = params.indicator[idx]
            if len(S) < x:
                continue

        mu_e = np.mean(S, axis=0)

        print("Quadratic filter.")
        x = len(S)
        p2 = copy.copy(params)
        p2.S = S
        p2.indicator = indicator
        p2.m = x
        
        idx = filter_c_sp(p2, M_mask, mu_e, fdr = fdr, plot=plotc, f=f)
        f+=1

        S =  p2.S[idx]
        indicator = p2.indicator[idx]
        m = len(S)

        if x == m: 
            print("Quadratic filter did not filter anything.")
            return topk_abs(mu_e, k)
        
def ransacGaussianMean(params):
    k = params.k
    d = params.d
    m = params.m
    eps = params.eps
    tau = params.tau
    S = params.S

    T_naive = np.sqrt(2*np.log(m*d/tau))
   
    med = np.median(S, axis=0)
    S = S[np.max(np.abs(med-S), axis=1) < T_naive]

    empmean = np.mean(S, axis=0)

    ransacN = S.shape[0]//2
    print("ransacN", ransacN)
    
    if ransacN > m: 
        return topk_abs(empmean, k)
    
    numIters = 5
    thresh = d + 2*(np.sqrt(d * np.log(m/tau)) + np.log(m/tau)) + (eps**2)*(np.log(1/eps))**2
    
    bestMean = empmean
    bestInliers = (S[LA.norm(S-empmean) < np.sqrt(thresh)]).shape[0]
    
    for i in np.arange(1, numIters, 1):
        ransacS = S[np.random.choice(S.shape[0], ransacN, replace=False)]
        ransacMean = np.mean(ransacS)
        curInliers = (S[LA.norm(S-ransacMean) < np.sqrt(thresh)]).shape[0]
        if curInliers > bestInliers:
            bestMean = ransacMean
            bestInliers = curInliers

    return topk_abs(bestMean, k)
