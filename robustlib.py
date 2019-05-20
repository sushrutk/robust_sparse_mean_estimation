import numpy as np
import scipy
import copy
from scipy import special
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mpld3
mpld3.enable_notebook()

class RunCollection(object):
    def __init__(self, func, inp):
        self.runs = []
        self.func = func
        self.inp = inp

    def run(self, trials):
        for i in range(trials):
            self.runs.append(self.func(*self.inp))


def get_error(f, params, tm):
    return LA.norm(tm - f(params))

""" Parameter object for noise models """

class Params(object):
    def __init__(self, d = 0, m = 0, eps = 0, k = 0, tau = 0.2):
        self.d = d
        self.m = m
        self.eps = eps
        self.k = k
        self.tau = tau


""" Noise models """

class DenseNoiseModel(object):
    def __init__(self, dist):
        self.dist = dist

    def generate(self, params):
        d, k, eps, m, tau = params.d, params.k, params.eps, params.m, params.tau

        tm = np.append(np.ones(k), np.zeros(d-k))

        G = np.random.randn(m, d) + tm

        S = G.copy()

        L = int(m*(1-eps))
        S[L:] += self.dist

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau)
        return params, S, indicator, tm



class BimodalModel(object):
    def __init__(self):
        pass
    
    def generate(self, params):
        d, k, eps, m, tau = params.d, params.k, params.eps, params.m, params.tau

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
        params = Params(d,m,eps,k,tau)
        return params, S, indicator, tm

class TailFilpModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        eps, m, d, k = params.eps, params.m, params.d, params.k

        tm = 0
        G = np.random.randn(m, d)
        S = G.copy()

        v = np.random.randn(k)
        v = np.append(v, np.zeros(d-k))
        np.random.shuffle(v)

        Sdots = S.dot(v)
        idx = np.argsort(Sdots)
        S = S[idx]

        S[:int(eps*m)] = -S[:int(eps*m)]
        
        indicator = np.ones(m)
        indicator[:int(eps*m)] = 0

        return params, S, indicator, tm


        
""" Algorithms """

class FilterAlgs(object):
    do_plot_linear = False
    do_plot_quadratic = False

    qfilter = True
    lfilter = True

    verbose = True

    is_sparse = True
    dense_filter = False
    
    figure_no = 0
    
    fdr = 0.1
    
    
    def __init__(self, params):
        self.params = params
        pass
    
    
    """ Tail estimates """
    
    def drop_points(self, S, indicator, x, tail, plot = False, f = 0):
        eps = self.params.eps
        m = self.params.m
        d = self.params.d
        fdr = self.fdr

        l = len(S)
        p_x = tail(x)
        p_x[p_x > 1] = 1

        sorted_idx = np.argsort(p_x)
        sorted_p = p_x[sorted_idx]

        T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
        if T == l : T = 0

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
            plt.title("sample size {}, T = {}, True FDR = {}, tail = {}".format(l, T, tfdr, tail.__name__))
            plt.xlabel("Experiments")
            plt.ylabel("p-values")
            plt.figure(f)

        return idx

    def tail_m(self, T):
        
        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau 
        
        return (special.erfc(T/np.sqrt(2)) + (eps**2)/(np.log(k*np.log(m*d/tau))*T**2))

    def tail_c(self, T): 
        
        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau    

        # idx = np.nonzero((T < 6))
        v = 3*np.exp(-T/3) + (eps**2/(T*(np.log(T)**2)))
        # v[idx] = 1
        
        return v

    def tail_t(self, T):
        """
        True tail.
        """

        eps, k, m, d, tau = self.params.eps, self.params.k, self.params.m, self.params.d, self.params.tau 

        return special.erfc(T/np.sqrt(2))


    def linear_filter(self, S, indicator, ev, v, u):
        eps = self.params.eps
        
        if ev > 1 + eps*np.sqrt(np.log(1/eps)): 
            if self.verbose:
                print("Linear filter...")
            l = len(S)
            S_u = S[np.ix_(np.arange(l),u)]
            dots = S_u.dot(v)
            m2 = np.median(dots)

            if self.dense_filter == False:
                x = np.abs(dots - m2) - 2*np.sqrt(ev*eps)
                idx = self.drop_points(S, indicator, x, self.tail_m,  self.do_plot_linear, self.figure_no)  
            else:
                x = np.abs(dots - m2)
                idx = self.drop_points(S, indicator, x, self.tail_t,  self.do_plot_linear, self.figure_no)  
            
            if self.verbose:
                bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
                print(f"Filtered out {l - len(idx[0])}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx[0])):0.2f} vs {self.fdr})")
            return idx
        else:
            return np.arange(len(S))
    
    def quadratic_filter(self, S, indicator, M_mask):

        print("Quadratic filter...")
        l = len(indicator)
        mu_e = np.mean(S, axis = 0)
        x = np.abs(p(S, mu_e, M_mask))

        idx = self.drop_points(S, indicator, x, self.tail_c, self.do_plot_quadratic, self.figure_no)
        
        if self.verbose:
            bad_filtered = np.sum(indicator) - np.sum(indicator[idx])
            print(f"Filtered out {l - len(idx[0])}/{l}, {bad_filtered} false ({bad_filtered / (l - len(idx[0])):0.2f} vs {self.fdr})")
            return idx
        else:
            return np.arange(len(S))
    
    def update_params(self, S, indicator, idx):
        S, indicator = S[idx], indicator[idx]
        self.params.m = len(idx)
        return S, indicator
    
    def alg(self, S, indicator):
        
        k = self.params.k
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        tau = self.params.tau
        
        T_naive = np.sqrt(2*np.log(m*d/tau))
        med = np.median(S, axis=0)
        idx = (np.max(np.abs(med-S), axis=1) < T_naive)
        S, indicator =  self.update_params(S, indicator, idx)

        if len(idx) < self.params.m: print("NP pruned {self.params.m - len(idx) f} points")
        
        while True:
            if self.lfilter == False and self.qfilter == False:
                break        

            if len(S)==0: 
                print("No points remaining.")
                return 0

            if len(S)==1: 
                print("1 point remaining.")
                return 0

            cov_e = np.cov(S, rowvar=0)
            M = cov_e - np.identity(d) 
            (mask, u) = indicat(M, k)
            M_mask = mask*M

            if self.dense_filter == False:
                if LA.norm(M_mask) < eps*(np.log(1/eps)): 
                    print("Valid output")
                    break

            if self.lfilter == True:

                if self.dense_filter == False:
                    cov_u = cov_e[np.ix_(u,u)]
                    ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1,k-1))
                    v = v.reshape(len(v),)
                else:
                    ev, v = scipy.linalg.eigh(cov_e, eigvals=(d-1,d-1))
                    if ev <  1+eps*np.log(1/eps):
                        print("RME exited properly")
                        break
                    v = v.reshape(len(v),)
                    u = np.arange(d)

                x = self.params.m
                idx = self.linear_filter(S, indicator, ev, v, u)
                self.figure_no += 1
                S, indicator =  self.update_params(S, indicator, idx)
                if len(idx) < x: continue

            if self.qfilter == True:

                x = self.params.m
                idx = self.quadratic_filter(S, indicator, M_mask)
                self.figure_no += 1
                S, indicator =  self.update_params(S, indicator, idx)
                if len(idx) < x: continue

            if x == len(idx): 
                print("Could not filter")
                break
                    
        if self.is_sparse == True:
            # print(topk_abs(np.mean(S, axis=0), k))
            return topk_abs(np.mean(S, axis=0), k)
        else:
            return np.mean(S, axis=0)
            
class NP_sp(FilterAlgs):
    lfilter, qfilter = False, False

class RME_sp(FilterAlgs):
    lfilter, qfilter = True, True

class RME_sp_L(FilterAlgs):
    lfilter, qfilter = True, False

class RME(FilterAlgs):
    
    lfilter, qfilter = True, False
    dense_filter = True
    # do_plot_linear = True



class ransacGaussianMean(object):
    def __init__(self, params):
        self.params = params
        pass

    def alg(self, S, indicator):
        k = self.params.k
        d = self.params.d
        m = self.params.m
        eps = self.params.eps
        tau = self.params.tau

        T_naive = np.sqrt(2*np.log(m*d/tau))
       
        med = np.median(S, axis=0)
        S = S[np.max(np.abs(med-S), axis=1) < T_naive]

        empmean = np.mean(S, axis=0)

        ransacN = S.shape[0]//2
        print("ransacN", ransacN)
        
        if ransacN > m: 
            return topk_abs(empmean, k)
        
        numIters = 5
        thresh = k*np.log(d) + 2*(np.sqrt(k* np.log(d) * np.log(m/tau)) + np.log(m/tau)) + (eps**2)*(np.log(1/eps))**2
        
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


class plot(RunCollection):

    def __init__(self, model, params, loss, keys = []):
        self.params = params
        self.keys = keys
        self.model = model
        self.loss = loss
        self.inp = 0
        self.Run = 0

    def get_dataxy(self, xvar_name, bounds, y_is_m = False, mrange = 0):
        results = {}

        for xvar in np.arange(*bounds):
            if xvar_name == 'm':
                self.params.m = xvar
            elif xvar_name == 'k':
                self.params.k = xvar
            elif xvar_name == 'd':
                self.params.d = xvar
            elif xvar_name == 'eps':
                self.params.eps = xvar

            if y_is_m == False:

                inp, S, indicator, tm = self.model.generate(self.params)

                # O = self.loss(topk_abs(np.mean(S * indicator[...,np.newaxis], axis=0), self.params.k), tm)
                O = self.loss(topk_abs(np.mean(S[:int(self.params.m*(1-self.params.eps))], axis=0), self.params.k), tm)
                
                for f in self.keys:
                    inp_copy = copy.copy(inp)
                    S_copy = S.copy()
                    indicator_copy = indicator.copy()

                    func = f(inp_copy)

                    results.setdefault(f.__name__, []).append(self.loss(func.alg(S_copy, indicator_copy), tm))
                    
                results.setdefault('oracle', []).append(O)

            else:

                l, s = mrange
                samp = l

                for f in self.keys:
                    while True:

                        count = 0
                        for i in range(10):

                            self.params.m = samp
                            inp, S, indicator, tm = self.model.generate(self.params)

                            func = f(inp)
                            vnew = self.loss(func.alg(S, indicator), tm)
                            print("VNEW ",vnew,"m ",samp,"xvar ",xvar,"count",count)
                        
                            # if vnew < 2*self.params.eps:
                            if vnew < 1.2:
                                count += 1
                        if count > 7:
                            break

                        samp += s
        
                    results.setdefault(f.__name__, []).append(samp)
                results.setdefault('oracle', []).append(O)
               
        return results

 
    def plot_xloss(self, Run, xvar_name, bounds):
        cols = {'RME_sp':'b', 'RME_sp_L':'g', 'RME':'r','ransacGaussianMean':'y' , 'NP_sp':'k', 'oracle':'c'}
        s = len(Run.runs)
        str_keys = [key.__name__ for key in self.keys]+['oracle']
        for key in str_keys:
            A = np.array([res[key] for res in Run.runs])
            xs = np.arange(*bounds)
            mins = [np.sort(x)[int(s*0.25)] for x in A.T]
            maxs = [np.sort(x)[int(s*0.75)] for x in A.T]

            plt.fill_between(xs, mins, maxs, color = cols[key], alpha=0.2)
            plt.plot(xs, np.median(A,axis = 0), label=key, color = cols[key])

        plt.title(f'd = {self.params.d}, k = {self.params.k}, eps = {self.params.eps}, m = {self.params.m}')
        plt.xlabel(xvar_name)
        plt.ylabel('MSE')
        plt.legend()


    def setdata(self, xvar_name, bounds, trials, ylims, y_is_m = False, mrange = []):
        Runs_l_samples = RunCollection(self.get_dataxy, (xvar_name, bounds, y_is_m, mrange))
        Runs_l_samples.run(trials)
        self.Run = Runs_l_samples


    def plotxy(self, xvar_name, bounds, ylims):
        self.plot_xloss(self.Run, xvar_name, bounds)
        plt.ylim(*ylims)
        plt.figure()


""" P(x) for quadratic filter """

def p(X, mu, M):
    F = LA.norm(M)
    D = X - mu
    vec = (D.dot(M) * D).sum(axis=1)
    return (vec - np.trace(M))/F


""" Thing that thresholds to the largest k entries indicaors """

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
        