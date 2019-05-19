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
        
class FilterAlgs(object):
    do_plot_linear = True
    do_plot_quadratic = True
    qfilter = True
    lfilter = True
    verbose = True
    figure_no = 0
    fdr = 0.1
    
    
    def __init__(self, params):
        self.params = params
        pass
    
    
    
    """ Tail estimates """
    
    def drop_points(self, params, x, tail, fdr = 0.1, plot = False, f = 0):
        S = params.S
        indicator = params.indicator
        eps = params.eps
        m = params.m
        d = params.d

        l = len(S)
        p_x = tail(x, params)
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

        if plot==True:
            plt.plot(np.arange(l), sorted_p)
            plt.plot(T*np.ones(100), 0.01*np.arange(100), linewidth=3)
            plt.plot(np.arange(l), indicator[sorted_idx], linestyle='-.', linewidth=3)
            plt.plot([0,len(S)],[0,fdr], '--')
            plt.title("sample size {}, T = {}, True FDR = {}, tail = {}".format(m, T, tfdr, tail.__name__))
            plt.xlabel("Experiments")
            plt.ylabel("p-values")
            plt.figure(f)

        return idx

    def tail_m(self, T):
        
        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau 
        
        return (special.erfc(T/np.sqrt(2)) + (eps**2)/(np.log(k*np.log(m*d/tau))*T**2))

    def tail_c(self, T): 
        
        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau    

        idx = np.nonzero((T < 6))
        v = 3*np.exp(-T/3) + (eps**2/(T*(np.log(T)**2)))
        v[idx] = 1
        
        return v

    def linear_filter(self, ev, v, u):
        S = self.params.S
        indicator = self.params.indicator
        eps = self.params.eps
        
        if ev > 1 + eps*np.sqrt(np.log(1/eps)): 
            if verbose:
                print("Linear filter...")

            x = len(S)
            
            p2 = copy.copy(self.params)
            p2.S = S[np.ix_(np.arange(x),u)]
            p2.indicator = indicator
            
            dots = p2.S.dot(v)
            m2 = np.median(dots)
            x = np.abs(dots - m2) - 3*np.sqrt(eps*ev)
            
            idx = self.drop_points(p2, x, tail_m, self.fdr, self.do_plot_linear, self.figure_no)  
            
            if verbose:
                bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
                print(f"Filtered out {x - len(idx[0])}/{x}, {bad_filtered} false ({bad_filtered / (x - len(idx[0])):0.2f} vs {fdr})")
            return idx
        else:
            return np.arange(len(S), 1)
    
    def quadratic_filter(self, M_mask): 
        mu_e = np.mean(self.params.S, axis = 0)
        p2 = copy.copy(self.params)        
        p_x = tail_c(np.abs(p(self.params.S, mu_e, M_mask)), p2)
        x = np.abs(p(self.params.S, mu_e, M_mask))

        idx = self.drop_points(p2, x, tail_c, self.fdr, self.do_plot_quadratic, self.figure_no)
                
        return idx
    
    def update_params(self, idx):
        self.params.S = self.params.S[idx]
        self.params.indicator = self.params.indicator[idx]
        self.params.m = len(idx)
        return None
    
    def alg(self):
        
        k = self.params.k
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        tau = self.params.tau
        
        T_naive = np.sqrt(2*np.log(m*d/tau))
        med = np.median(self.params.S, axis=0)
        idx = (np.max(np.abs(med-self.params.S), axis=1) < T_naive)
        if len(idx) < self.params.m: print("NP pruned {self.params.m - len(idx) f} points")
        
        while True:
            if self.lfilter == False and self.qfilter == False:
                break        

            if len(self.params.S)==0: 
                print("No points remaining.")
                return None

            if len(self.params.S)==1: 
                print("1 point remaining.")
                return None

            cov_e = np.cov(self.params.S, rowvar=0)
            M = cov_e - np.identity(d) 
            (mask, u) = indicat(M, k)
            M_mask = mask*M

            if LA.norm(M_mask) < eps*np.log(1/eps): 
                print("Valid output")
                break

            if self.lfilter == True:

                cov_u = cov_e[np.ix_(u,u)]
                ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1,k-1))
                v = v.reshape(len(v),)

                x = self.params.m
                idx = self.linear_filter(ev, v, u)
                self.figure_no += 1
                self.update_params(idx)
                if len(idx) < x: continue

            if self.qfilter == True:

                x = self.params.m
                idx = self.quadratic_filter(M_mask)
                self.figure_no += 1
                self.update_params(idx)
                if len(idx) < x: continue

            if x == len(idx): 
                print("Neither filter filtered anything.")
                break
                    
        if is_sparse == True:
            return topk_abs(np.mean(self.params.S, axis=0))
        else:
            return np.mean(self.params.S, axis=0)
            
        


def sparse_samp_loss(model_params, keys, m_bounds):
    (Low, Up, step) = m_bounds
    
    results = {}

    for m in np.arange(Low, Up, step):
        d, k, eps = model_params
        model = BimodalModel()
        params, tm = model.generate(d, k, eps, m)
        
        O = LA.norm(tm - np.mean(params.S * params.indicator[...,np.newaxis], axis=0))
        
        for f in keys:
            results.setdefault(f.__name__, []).append(LA.norm(tm - f(params))/eps)
            
        results.setdefault('oracle', []).append(O/eps)
        results.setdefault('eps', []).append(1)
    
    return results

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


""" Filters """


""" Auxillary functions """

""" P(x) for quadratic filter """

def p(X, mu, M):
    F = LA.norm(M)
    D = X - mu
    vec = (D.dot(M) * D).sum(axis=1)
    return (vec - np.trace(M))/F



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
