import numpy as np
import scipy
from scipy import special
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mpld3
mpld3.enable_notebook()




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


"""
Defining the tail probabilities, however we can just simulate the correct
distribution, potentially.
"""

def tail_m(T, eps, k, m, d, tau):
    return (3*special.erfc(T/np.sqrt(2)) + (eps**2)/(np.log(k*np.log(m*d/tau))*T**2))

def tail_c(T, eps, k, m, d, tau): 
    
    idx = np.nonzero((T < 6))
    v = 3*np.exp(-T/3) + (eps**2/(T*(np.log(T)**2)))
    v[idx] = 1
    return v

def p(X, mu, M):
    F = LA.norm(M)
    D = X - mu
    vec = (D.dot(M) * D).sum(axis=1)
    return (vec - np.trace(M.T.dot(M)))/F

"""
Defining filters.
"""

def filter_m_sp(S, indicator, v, ev, u, eps, k, tau, fdr = 0.1, plot=0, f=0):
    
    """This filters elements of S whenever the 
    deviation of <x, v> from the median 
    is more than reasonable. 
    """
    m, d = S.shape
    l = len(S)
    dots = S.dot(v)
    m2 = np.median(dots)

    x = np.abs(dots - m2) - 3 np.sqrt(eps*ev)
    p_x = tail_m(np.abs(dots - m2), eps, k, m, d, tau)
    p_x[p_x > 1] = 1
    
    sorted_idx = np.argsort(p_x)
    sorted_p = p_x[sorted_idx]

        
    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
    if T>0.5*m: T = 0
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


def filter_c_sp(S, indicator, M_u, mu, eps, k, tau, u, fdr=0.1, plot =0, f=0):
    
    """
    This filters elements of S whenever the 
    degree 2 polynomial p(X) is larger
    than reasonable.
    """
    
    l = len(S)
    m, d = S.shape
    p_x = tail_c(np.abs(p(S, mu, M_u)), eps, k, m, d, tau)
    x = np.abs(p(S, mu, M_u))

    p_x[p_x > 1] = 1
    sorted_idx = np.argsort(p_x)
    sorted_p = p_x[sorted_idx]
    print(sorted_p)
#     sorted_p[sorted_p == 1] = 0
#     print(p_x)

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

def NP_sp(params):
    S, indicator, k, eps, tau = params
    m, d = S.shape
    T_naive = np.sqrt(np.log(m*d/tau))
   
    med = np.median(S, axis=0)
    S = S[np.max(np.abs(med-S), axis=1) < T_naive]
    return(S)

def NP_sp_mean(params):
    S = NP_sp(params)
    if m > len(S):
        print("NP_sp pruned!")
    mean = np.mean(S, axis=0)
#    print(mean)
    u = np.argpartition(mean.abs(), -k)[-k:]
    z = np.zeros(len(mean))
    z[u] = mean[u]
    return z

def RME_sp(params, plotl = 0, plotc = 0, f = 0, fdr=0.2):
    S, indicator, k, eps, tau = params
    m, d = S.shape
    
    if plotl ==1 or plotc==1:
        print("Total bad points: ", np.sum(1-indicator))
    T_naive = np.sqrt(np.log(m*d/tau))
   
    med = np.median(S, axis=0)
    idx = (np.max(np.abs(med-S), axis=1) < T_naive)
    # S = S[np.max(np.abs(med-S), axis=1) < T_naive]
    S =  S[idx]
    indicator = indicator[idx]
    if plotl==1 or plotc==1:
        print("Bad points after NP: ", np.sum(1-indicator))


    mu_e = np.mean(S, axis=0) 
#    print(np.mean(S, axis=0))

    while True:
        
        if len(S)==0: 
            print("No points remaining.")
            u2 = np.argpartition(mu_e, -k)[-k:]
#            print(u)
            z = np.zeros(len(mu_e))
            z[u2] = mu_e[u2]
            return z

        if len(S)==1: 
            print("1 point remaining.")
            u2 = np.argpartition(mu_e, -k)[-k:]
#            print(u)
            z = np.zeros(len(mu_e))
            z[u2] = mu_e[u2]
            return z


        
        mu_e = np.mean(S, axis=0) 
        cov_e = np.cov(S, rowvar=0)
        
        M = cov_e - np.identity(d) 
        (mask, u) = indicat(M, k)
#        print(u)
#        print(mu_e[u])
#        print(M)
        M_mask = mask*M
#        print(LA.norm(M_mask))

        if LA.norm(M_mask) < eps*np.log(1/eps): 
            print("Valid output")
            u2 = np.argpartition(mu_e, -k)[-k:]
            z = np.zeros(len(mu_e))
            z[u2] = mu_e[u2]
            return z

        cov_u = cov_e[np.ix_(u,u)]
        ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1,k-1))
        v = v.reshape(len(v),)
                
        if ev > 1 + eps*np.sqrt(np.log(1/eps)): 

            print("Linear filter.")

            x = len(S)
#                print(x)
            S2 = S[np.ix_(np.arange(len(S)),u)]
#                print(S2.shape)
            idx = filter_m_sp(S2, indicator, v, u, eps, k, x, d, tau, fdr = fdr, plot=plotl, f=f)
            f+=1
#                print(len(idx))
#                print(idx)
            S =  S[idx]
            indicator = indicator[idx]
#                print(len(S))

            if x == len(S): 
                print("Linear filter did not filter anything.")
#                    return mu_e

        print("Quadratic filter.")
        x = len(S)
#        print(x)
        idx = filter_c_sp(S, indicator, M_mask, mu_e, eps, k, x, d, tau, u, fdr = fdr, plot=plotc, f=f)
        f+=1
#        print(len(idx[0]))
#        print(idx)

        S =  S[idx]
        indicator = indicator[idx]

#        print(len(S))

        if x == len(S): 
            print("Quadratic filter did not filter anything.")
            u2 = np.argpartition(mu_e, -k)[-k:]
#            print(u)
            z = np.zeros(len(mu_e))
            z[u2] = mu_e[u2]
            return z


def RME_sp_L(params, plotl = 0, plotc = 0, f = 0, fdr=0.2):
    S, indicator, k, eps, tau = params
    m, d = S.shape
    
    if plotl ==1 or plotc==1:
        print("Total bad points: ", np.sum(1-indicator))
    T_naive = np.sqrt(np.log(m*d/tau))
   
    med = np.median(S, axis=0)
    idx = (np.max(np.abs(med-S), axis=1) < T_naive)
    # S = S[np.max(np.abs(med-S), axis=1) < T_naive]
    S =  S[idx]
    indicator = indicator[idx]
    if plotl==1 or plotc==1:
        print("Bad points after NP: ", np.sum(1-indicator))


    mu_e = np.mean(S, axis=0) 
#    print(np.mean(S, axis=0))

    while True:
        
        if len(S)==0: 
            print("No points remaining.")
            u2 = np.argpartition(mu_e, -k)[-k:]
#            print(u)
            z = np.zeros(len(mu_e))
            z[u2] = mu_e[u2]
            return z

        if len(S)==1: 
            print("1 point remaining.")
            u2 = np.argpartition(mu_e, -k)[-k:]
#            print(u)
            z = np.zeros(len(mu_e))
            z[u2] = mu_e[u2]
            return z


        
        mu_e = np.mean(S, axis=0) 
        print(len(S))

        cov_e = np.cov(S, rowvar=0)
        
        M = cov_e - np.identity(d) 
        (mask, u) = indicat(M, k)
#        print(u)
#        print(mu_e[u])
#        print(M)
        M_mask = mask*M
#        print(LA.norm(M_mask))

        # if LA.norm(M_mask) < eps*np.log(1/eps): 
        #     print("Valid output")
        #     u2 = np.argpartition(mu_e, -k)[-k:]
        #     z = np.zeros(len(mu_e))
        #     z[u2] = mu_e[u2]
        #     return z

        cov_u = cov_e[np.ix_(u,u)]
        ev, v = scipy.linalg.eigh(cov_u, eigvals=(k-1,k-1))
        v = v.reshape(len(v),)
                
        if ev > 1 + eps*np.sqrt(np.log(1/eps)): 

            print("Linear filter.")
            x = len(S)
#                print(x)
            S2 = S[np.ix_(np.arange(len(S)),u)]
#                print(S2.shape)
            idx = filter_m_sp(S2, indicator, v, u, eps, k, x, d, tau, fdr = fdr, plot=plotl, f=f)
            f+=1
#                print(len(idx))
#                print(idx)
            S =  S[idx]
            indicator = indicator[idx]
#                print(len(S))

            if x == len(S): 
                print("Linear filter did not filter anything.")
                u2 = np.argpartition(mu_e, -k)[-k:]
    #            print(u)
                z = np.zeros(len(mu_e))
                z[u2] = mu_e[u2]
                return z
        else:
            u2 = np.argpartition(mu_e, -k)[-k:]
#            print(u)
            z = np.zeros(len(mu_e))
            z[u2] = mu_e[u2]
            return z
#         print("Quadratic filter.")
#         x = len(S)
# #        print(x)
#         idx = filter_c_sp(S, indicator, M_mask, mu_e, eps, k, x, d, tau, u, fdr = fdr, plot=plotc, f=f)
#         f+=1
# #        print(len(idx[0]))
# #        print(idx)

#         S =  S[idx]
#         indicator = indicator[idx]

# #        print(len(S))

#         if x == len(S): 
#             print("Quadratic filter did not filter anything.")
#             u2 = np.argpartition(mu_e, -k)[-k:]
# #            print(u)
#             z = np.zeros(len(mu_e))
#             z[u2] = mu_e[u2]
#             return z



def ransacGaussianMean(params):
    S, indicator, k, eps, tau = params
    m, d = S.shape
    
    T_naive = np.sqrt(np.log(m*d/tau))
   
    med = np.median(S, axis=0)
    S = S[np.max(np.abs(med-S), axis=1) < T_naive]

#     print(m, d)
    empmean = np.mean(S, axis=0)
#     print(empmean)
    ransacN = int(2*(d*np.log(4) + np.log(2/tau))/eps**2)
#     print("ransacN", ransacN)
    
    if ransacN > m: 
        u2 = np.argpartition(empmean, -k)[-k:]
        z = np.zeros(len(empmean))
        z[u2] = empmean[u2]
        return z
    
    numIters = 100
    thresh = d + 2*(np.sqrt(d * np.log(m/tau)) + np.log(m/tau)) + (eps**2)*(np.log(1/eps))**2
    
    bestMean = empmean
    bestInliers = (S[LA.norm(S-empmean) < np.sqrt(thresh)]).shape[0]
    
    for i in np.arange(1, numIters, 1):
        ransacS = S[np.random.choice(ransacN, S.shape[1], replace=False)]
        ransacMean = np.mean(ransacS)
        curInliers = (S[LA.norm(S-ransacMean) < np.sqrt(thresh)]).shape[0]
        if curInliers > bestInliers:
            bestMean = ransacMean
            bestInliers = curInliers
            
    u2 = np.argpartition(bestMean, -k)[-k:]
    z = np.zeros(len(bestMean))
    z[u2] = bestMean[u2]
    return z


def k_loss(params, bounds):
    (d, m, EPS, tau) = params
    (Low, Up, scale) = bounds
    
    results = {}

    for eps in EPS:        
        for k in scale*np.arange(Low,Up):

#             true_mean = np.append(np.ones(k),np.zeros(d-k))
#             false_mean = np.append(np.zeros(d-k), np.ones(k))
#             if eps<0.001:
#                 m = 2000
#             else:
#                 m = int(4*((k**2)*np.log(d) + np.log(1/tau))*(1/eps**2))
                
            tm = np.append(np.ones(k), np.zeros(d-k))
            fm = np.append(np.zeros(d-k), np.ones(k))

            cov = 2*np.identity(d) - np.diag(tm)
            tm = tm/LA.norm(tm)

            G = np.random.randn(m, d) + tm
            G2 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
#             G1 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
            G1 = np.random.randn(m, d) + fm

            S = G.copy()

            indicator = np.zeros(len(S))
            L = int(m*(1-eps))
            M = int((L + m)/2)
            print(m-L)

            S[L:M] = G1[L:M]
            S[M:] = G2[M:]

            indicator = np.ones(len(S))
            indicator[int(m*(1-eps)):] = 0
            params = (S.copy(), indicator, k, eps, tau)

            mu_o = np.mean(S[:int(m*(1-eps))], axis=0)
            u2 = np.argpartition(mu_o, -k)[-k:]
            z = np.zeros(len(mu_o))
            z[u2] = mu_o[u2]

            O = LA.norm(tm - z)

            for key in ['RME_sp_L', 'RME_sp', 'RME', 'NP_sp', 'ransacGaussianMean']:
                params = (S.copy(), indicator, k, eps, tau)
                if key=='RME':
                    params =  (G.copy(), S.copy(), indicator, m, d, eps, tau)
                results.setdefault(key+str(eps), []).append(LA.norm(tm - eval(key+'(params)'))/eps)

            results.setdefault('oracle'+str(eps), []).append(O/eps)
            results.setdefault('eps'+str(eps), []).append(1)
    
    return results

def sparse_eps(params, bounds):
    (k, m, eps, tau) = params
    (Low, Up, scale) = bounds
    
    results = {}

    for d in scale*np.arange(Low,Up):
        
        true_mean = np.append(np.ones(k),np.zeros(d-k))
#        true_mean = 0
#         m = int(4*((k**2)*np.log(d) + np.log(1/tau))*(1/eps**2))
        G = np.random.randn(m, d) + true_mean

        S = G.copy()
        S[int(m*(1-eps)):] = true_mean + 4
        indicator = np.ones(len(S))
        indicator[int(m*(1-eps)):] = 0
        params = (S.copy(), indicator, k, eps, tau)

       
        mu_o = np.mean(S[:int(m*(1-eps))], axis=0)
        u2 = np.argpartition(mu_o, -k)[-k:]
        z = np.zeros(len(mu_o))
        z[u2] = mu_o[u2]

        O = LA.norm(true_mean - z)

        for key in ['RME_sp', 'NP_sp', 'RME_sp_L']:
            params = (S.copy(), indicator, k, eps, tau)
            if key=='RME':
                params =  (G.copy(), S.copy(), indicator, m, d, eps, tau)
            results.setdefault(key, []).append(LA.norm(true_mean - eval(key+'(params)'))/eps)

        results.setdefault('oracle', []).append(O/eps)
        results.setdefault('eps', []).append(1)
    
    return results

def sparse_eps_dist(params, bounds):
    (d, k, m, eps, tau) = params
    (Low, Up, step) = bounds
    
    results = {}

    for dist in np.arange(Low, Up, step):
        
        true_mean = np.append(np.ones(k),np.zeros(d-k))
#         false_mean = np.append(np.zeros(d-k), np.ones(k))
        false_mean = np.ones(d)

#        true_mean = 0
        G = np.random.randn(m, d) + true_mean

        S = G.copy()
#         S[int(m*(1-eps)):] = true_mean + dist
        S[int(m*(1-eps)):] += dist*false_mean + true_mean
#         S[int(m*(1-eps)):] = np.sqrt(np.log(m*d/tau))-0.1
        indicator = np.ones(len(S))
        indicator[int(m*(1-eps)):] = 0
        params = (S.copy(), indicator, k, eps, tau)

        mu_o = np.mean(S[:int(m*(1-eps))], axis=0)
        u2 = np.argpartition(mu_o, -k)[-k:]
        z = np.zeros(len(mu_o))
        z[u2] = mu_o[u2]

        O = LA.norm(true_mean - z)
        
        for key in ['RME_sp', 'RME_sp_L', 'NP_sp', 'ransacGaussianMean', 'RME']:
            params = (S.copy(), indicator, k, eps, tau)
            if key=='RME':
                params =  (G.copy(), S.copy(), indicator, m, d, eps, tau)
            results.setdefault(key, []).append(LA.norm(true_mean - eval(key+'(params)'))/eps)

        results.setdefault('oracle', []).append(O/eps)
        results.setdefault('eps', []).append(1)
    
    return results

"""
Sample vs loss for noise at distance 0.8
"""

def sparse_samp_loss(params, bounds): 
    (d, k, dist, eps) = params
    (Low, Up, step) = bounds


    results = {}

    for samp in np.arange(Low, Up, step):
        
        m = samp
        
        tm = np.append(np.ones(k), np.zeros(d-k))
        fm = np.append(np.zeros(d-k), np.ones(k))

        cov = 2*np.identity(d) - np.diag(tm)
#         tm = tm/LA.norm(tm)

        G = np.random.randn(m, d) + tm
        G2 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
        G1 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
#         G1 = fm

        S = G.copy()

        indicator = np.zeros(len(S))
        L = int(m*(1-eps))
        M = int((L + m)/2)
        print(m-L)

#         S[L:M] = G1[L:M]
#         S[M:] = G2[M:]
        tau = 0.2
        S[L:] =  np.sqrt(np.log(m*d/tau))-0.1



        indicator = np.ones(len(S))
        indicator[int(m*(1-eps)):] = 0



#         tm = np.append(np.ones(k),np.zeros(d-k))
# #         fm = np.append(np.zeros(d-k), np.ones(k))
#         fm = np.ones(d)

#         G = np.random.randn(m, d) + tm

#         S = G.copy()
#         S[int(m*(1-eps)):] += dist*fm + tm
# #         S[int(m*(1-eps)):] =  dist
#         indicator = np.ones(len(S))
#         indicator[int(m*(1-eps)):] = 0
        params = (S.copy(), indicator, k, eps, 0.2)



        O = LA.norm(tm - np.mean(S[:int(m*(1-eps))], axis=0))

        # for func in [RME_sp, NP_sp]:
        # ... func(params)
        # Option 1: set RME_sp.name = 'RME_sp' somewhere, use func.name
        # Option 2: func.__name__
        
        for f in [RME_sp_L, RME_sp, RME, NP_sp, ransacGaussianMean]:
            params = (S.copy(), indicator, k, eps, 0.2)
            if f == RME:
                params = (G.copy(), S.copy(), indicator, m, d, eps, 0.2)
            if f == RME:
                mu_o = f(params)
                if mu_o.__class__.__name__ != 'int':
                    u2 = np.argpartition(mu_o, -k)[-k:]
                    z = np.zeros(len(mu_o))
                    z[u2] = mu_o[u2]
                    results.setdefault(f.__name__, []).append(LA.norm(tm - z)/eps)
                else:
                    results.setdefault(f.__name__, []).append(LA.norm(tm - f(params))/eps)
                    
            else: 
                results.setdefault(f.__name__, []).append(LA.norm(tm - f(params))/eps)

#         for f in [rl.RME_sp_L, rl.NP_sp]:
#             results.setdefault(f.__name__, []).append(LA.norm(true_mean - f(params))/eps)

        results.setdefault('oracle', []).append(O/eps)
        results.setdefault('eps', []).append(1)
    
    return results

"""
Sample vs loss for noise at distance d
"""

def sparse_sampcomplexity_k(params, bounds): 
    (d, dist, eps, tau, maxk) = params
    (low, step) = bounds

    results = np.array([])
    m = low

    for k in np.arange(2,maxk+1):
        dev = 1000
        temp = np.array([0])
        count = 0
        val0 = 0
        if m > step: m -= step
#         if m > 10: m -=10
        while True:
            print(m)
            count = 0
            for i in range(10):
#                 true_mean = np.append(np.ones(k),np.zeros(d-k))
#                 true_mean = 0
    
#                 tm = np.append(np.ones(k), np.zeros(d-k))
#                 fm = np.append(np.zeros(d-k), np.ones(k))

#                 cov = 2*np.identity(d) - np.diag(tm)
#                 tm = tm/LA.norm(tm)

#                 G = np.random.randn(m, d) + tm
#                 G2 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
#     #             G1 = np.random.multivariate_normal(np.zeros(d), cov, (m,))
#                 G1 = np.random.randn(m, d) + fm

#                 S = G.copy()

#                 indicator = np.zeros(len(S))
#                 L = int(m*(1-eps))
#                 M = int((L + m)/2)
# #                 print(m-L)

#                 S[L:M] = G1[L:M]
#                 S[M:] = G2[M:]

                G = np.random.randn(m, d)
            
# #                 V = [np.append(np.zeros(d-k), np.ones(k)) for i in range(int(m*(1-eps)), m)]
# #                 for v in V:
# #                     np.random.shuffle(v)
# #                 N = eps*np.array(V)*(1/np.sqrt(k))

                v0 = np.random.randn(k); v0 = v0/LA.norm(v0)
                v = np.append(np.zeros(d-k), v0)
                np.random.shuffle(v)
                u = v.copy()
                np.random.shuffle(u)
              
                S = G.copy()
#                 S[int(m*(1-eps)):] = N
                L = int(m*(1-eps))
                M = int((L+m)/2)
                S[L:M] += (eps)*v
                S[M:m] -= (eps)*v 
#                 if np.random.randn(1) > 0:
                true_mean = 0
#                 else:
#                     true_mean = eps*v

                
                indicator = np.ones(len(S))
                indicator[int(m*(1-eps)):]=0
                params = (S.copy(), indicator, k, eps, tau)


                # for func in [RME_sp, NP_sp]:
                # ... func(params)
                # Option 1: set RME_sp.name = 'RME_sp' somewhere, use func.name
                # Option 2: func.__name__
    #             if m==mprev:
    #                 val0 = LA.norm(true_mean - RME_sp(params))
                vnew = LA.norm(true_mean - RME_sp(params))
                temp = np.append(temp, vnew)
                print("VNEW ",vnew,"m ",m,"k ",k,"count",count)
            
                if vnew < 2*eps:
                    count+=1
            if count > 7:
                break
            m+=step


        #mprev = m
        
        results = np.append(results, m)
        
    return results

"""
Sample vs loss for noise at distance d
"""

def sparse_sampcomplexity_d(params, bounds): 
    (k, dist, eps, tau) = params
    (low, step) = bounds

    results = np.array([])
    m = low



    for d in np.arange(10,15):
        dev = 1000
        temp = np.array([0])
        count = 0
        val0 = 0
        while True:
            print(m)
            true_mean = np.append(np.ones(k),np.zeros(d-k))
    #        true_mean = 0
            G = np.random.randn(m, d) + true_mean

            S = G.copy()
            S[int(m*(1-eps)):] = true_mean + dist

            indicator = np.ones(len(S))
            indicator[int(m*(1-eps)):]=0
            params = (S.copy(), indicator, k, eps, tau)
            
            # for func in [RME_sp, NP_sp]:
            # ... func(params)
            # Option 1: set RME_sp.name = 'RME_sp' somewhere, use func.name
            # Option 2: func.__name__
#             if m==mprev:
#                 val0 = LA.norm(true_mean - RME_sp(params))
            vnew = LA.norm(true_mean - RME_sp(params))
            temp = np.append(temp, vnew)
            print(val0)
            print("VNEW "+str(vnew))
            
            if vnew < 1 :
                break
            m+=step


        #mprev = m
        
        results = np.append(results, m)
        
    return results



"-------.............................................-----------"

def p(X, mu, M):
    F = LA.norm(M)
    D = X - mu
    vec = (D.dot(M) * D).sum(axis=1)
    return (vec - np.trace(M.T.dot(M)))/F

def tail_t(x):
    """
    True tail.
    """
    return special.erfc(x/np.sqrt(2))


def e_tail_gauss(x,s,m):
    G = np.random.randn(s,m)
#    print(G.shape)
    p_x = (s - np.apply_along_axis(lambda a: np.sort(a).searchsorted(x), axis = 0, arr = G))/(s+0.0)
#    print(len(p_x))
    return 2*np.mean(p_x, axis=1)

"Do this thing for the Quadratic filter in the sparse algorithm"
def e_tail_sp_qgauss(x, M, d, k, s, m):
    G = np.random.randn(s, m, d)

    vals = []
    for i in range(s):
        M = np.cov(G[i],rowvar=False)-np.identity(d)
        mask, u = indicat(M, k)
        vals.append(p(G[i], np.mean(G[i]), M*mask))

    # vals = [p(G[i],np.mean(G[i]),np.cov(G[i],rowvar=False)-np.identity(d)) for i in range(s)]

    # vals = [p(G[i],np.mean(G[i]),M) for i in range(s)]
    p_x = (m - np.apply_along_axis(lambda a: np.sort(a).searchsorted(x), axis = 1, arr = vals))/(m+0.0)
    return np.mean(p_x, axis=0)

"Same emperical p-values for the filter in the robust sparse pca algorithm"
def e_tail_qgauss(x,v,s,m):
    G = np.random.randn(s,m)

#    print(G.shape)
    p_x = (s - np.apply_along_axis(lambda a: np.sort(a).searchsorted(x), axis = 0, arr = G))/(s+0.0)
#    print(len(p_x))
    return 2*np.mean(p_x, axis=1)


def tail_t2(x, eps):
    x-=3
    idx = np.nonzero((x < 10*np.log(1/eps)))
    v = (1/((x*np.log(x))**2))
    v = (eps)*v


    "What is the chance that things are smaller than 4"

    # v += 2*np.exp(-x)
    # v[v>1]=1
    v[idx] = 1

    # print(x)
    # v = 2*np.exp(-x)
    # print(v)
    # v[v>1]=1

    return v

# def tail_m(x, m, d, eps, tau):
#     """
#     Upper bound from writeup.
#     """
#     return 2*(special.erfc(x/np.sqrt(2))+(eps**2)/(np.log(d*np.log(m*d/tau))*x**2))


def filter_m_rspca(params, f=0, plot=0, fdr = .5):
    
    """
    This filters elements of S whenever the 
    deviation of <x, v> from the median 
    is more than reasonable. 
    """ 
    (S, indicator, v, m, d, eps) = params
    
    l = len(S)

    #dots = S.dot(v)
    #m2 = np.median(dots)    
    
 #   Gv = (G.dot(v)).reshape(m,)
 #   med_G = np.median(Gv)
 #   px_Gdev = np.sort(tail_t2(np.abs(Gv - med_G)))
 #   px_Gdev = (np.arange(l)/l).reshape(px_Gdev.shape) - px_Gdev
 #   px_Gdev[px_Gdev < 0] = 0

    Sv = (S.dot(v)).reshape(m,)
    med_S = np.median(Sv)
    idx = np.argsort(tail_t2(np.abs(Sv - med_S), eps))
    sidx = np.argsort(tail_t2(np.abs(Sv - med_S), eps))
    px_S = np.sort(tail_t2(np.abs(Sv - med_S), eps))
#    px_S += px_Gdev
    
    sorted_p = px_S
    sorted_p[sorted_p > 1] = 1
    sorted_p[np.isnan(sorted_p)] = 0
    sorted_p = np.sort(sorted_p)
    
    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
#    print("pre-T", np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1]))
#     if np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])==0:
#         print((sorted_p - (fdr/l)*np.arange(l) > 0))
#         print(sorted_p - (fdr/l)*np.arange(l))
#         print(sorted_p)
# #    print("T",T)
    if T==l : T = 0

    idx = idx[T:]
    
    if len(S)==len(idx):
        tfdr = 0
    else:
        tfdr = (sum(indicator) - sum(indicator[idx]))/(len(S)-len(idx))

    """
    p_x = tail_m(np.abs(dots - m2), k, m, d, eps, tau)
    p_x[p_x > 1] = 1
    idx = np.argsort(p_x)
    sorted_p = np.sort(p_x)
    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
    if T == l: T = 0
    """

    if plot==1:
        plt.plot(np.arange(l), sorted_p)
        plt.plot(T*np.ones(100), 0.01*np.arange(100), linewidth=3)
        plt.plot(np.arange(len(sidx)), indicator[sidx], linestyle='-.',linewidth=1)
        plt.plot([0,len(S)],[0,fdr], '--')
        plt.title("sample size {}, True FDR: {}, T = {}".format(m, tfdr, T))
        plt.figure(f)


    
    return idx


def filter_m(params, f=0, plot=0, fdr = .5, ind_p = 1000):
    
    """
    This filters elements of S whenever the 
    deviation of <x, v> from the median 
    is more than reasonable. 
    """ 
    (G, S, indicator, v, m, d, eps, tau) = params
    
    l = len(S)

    #dots = S.dot(v)
    #m2 = np.median(dots)    
    
    # Gv = G.dot(v)
    # med_G = np.median(Gv)
    # px_Gdev = np.sort(tail_t(np.abs(Gv - med_G)))
    # px_Gdev = (np.arange(l)/l).reshape(px_Gdev.shape) - px_Gdev
    # px_Gdev[px_Gdev < 0] = 0
    
    "Using tail_t"

    Sv = S.dot(v)
    med_S = np.median(Sv)
    idx = np.argsort(tail_t(np.abs(Sv - med_S)))
    sidx = np.argsort(tail_t(np.abs(Sv - med_S)))
    px_S = np.sort(tail_t(np.abs(Sv - med_S)))

    # px_S += px_Gdev
    
    "Using e_tail_gauss"

    # Sv = S.dot(v)
    # med_S = np.median(Sv)
    # idx = np.argsort(e_tail_gauss(np.abs(Sv - med_S), len(Sv), ind_p))
    # sidx = np.argsort(e_tail_gauss(np.abs(Sv - med_S), len(Sv), ind_p))
    # px_S = np.sort(e_tail_gauss(np.abs(Sv - med_S), len(Sv), ind_p))
    #plt.plot(np.arange(len(px_S)), px_S)



    sorted_p = px_S
    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1]) 
    if T==l : T = 0

    idx = idx[T:]


    if len(S)==len(idx):
        tfdr = 0
    else:
        tfdr = (sum(indicator) - sum(indicator[idx]))/(len(S)-len(idx))

    
    """
    p_x = tail_m(np.abs(dots - m2), k, m, d, eps, tau)
    p_x[p_x > 1] = 1
    idx = np.argsort(p_x)
    sorted_p = np.sort(p_x)
    T = l - np.argmin((sorted_p - (fdr/l)*np.arange(l) > 0)[::-1])
    if T == l: T = 0
    """

    if plot==1:
        plt.figure(f)
        plt.plot(np.arange(l), sorted_p, label = 'p-values')
        plt.plot(T*np.ones(100), 0.01*np.arange(100), label = 'T')
        plt.plot(np.arange(len(sidx)), indicator[sidx], linestyle='-.',linewidth=3, label='Good-indicator')
        plt.plot([0,len(S)],[0,fdr], '--')
        plt.legend()
        plt.xlabel("Samples")
        plt.ylabel("p-value")
        plt.title("Samples pruned in step {}, True FDR: {}".format(f+1,tfdr))

    
    return idx



def NP(S, m, d, tau):

    T_naive = np.sqrt(np.log(m*d/tau))
    med = np.median(S, axis=0)
    S = S[np.max(np.abs(med-S), axis=1) < T_naive]
    
    return S

def NPmean(params):
    (G, S, indicator, m, d, eps, tau) = params
    return np.mean(NP(S, m, d, tau), axis=0)

def Mean(params):
    (G, S, indicator, m, d, eps, tau) = params
    return np.mean(S, axis=0)


def RME(params, fdr = .2, plot=0, thresh = 2, ind_p = 1000):
    (G, S, indicator, m, d, eps, tau) = params
    if d!=1:
        T_naive = np.sqrt(np.log(m*d/tau))
        med = np.median(S, axis=0)
        idx = np.nonzero((np.max(np.abs(med-S), axis=1) < T_naive))
        S = S[idx]
        indicator = indicator[idx]


 #   G = G[idx]
    
    count = 0

    while True:
        if len(S)<=1:
            return 0
        
        mu_e = np.mean(S, axis=0) 
        cov_e = np.cov(S, rowvar=0) 

        M = cov_e - np.identity(d) 

        ev, v = scipy.linalg.eigh(M, eigvals=(d-1,d-1))
        v = v.reshape(len(v),)
        
        if ev < thresh * np.sqrt(np.log(1/eps)):
            return mu_e

        l = len(S)
        idx = filter_m((G, S, indicator, v, m, d, eps, tau), count, plot, fdr, ind_p)
#        print(idx)
 #       S, G = S[idx], G[idx]
        S = S[idx]
        indicator = indicator[idx]

        count+=1
        if len(S)==l:
            print("filter did not apply")
            return np.mean(S, axis=0) 

def RME_greedy(params, frac = 5, thresh = 10, plot = 0):
    (G, S, indicator, m, d, eps, tau) = params

    S = NP(S, m, d, tau)
    
    while True:
        
        if len(S)<=1:
            print("RME_greedy pruned too aggressively")
            return mu_e
        
        mu_e = np.mean(S, axis=0) 
        cov_e = np.cov(S, rowvar=0)
        M = cov_e - np.identity(d) 
        
        ev, v = scipy.linalg.eigh(M, eigvals=(d-1,d-1))
        v = v.reshape(len(v),)

        if ev < thresh * eps*np.sqrt(np.log(1/eps)):
            return mu_e

        vec = np.abs((S-mu_e).dot(v))
        vec = vec.reshape(len(vec),)
        
        idx = np.argsort(vec)
        vec = np.sort(vec)[::-1]
        
        variance = LA.norm(vec)**2 
        
        s = 0
        for i in range(len(vec)):
            if s <= variance / frac: 
                s+=vec[i]**2
            else: break
                
        l = len(S)
        S = S[idx[:len(vec)-i]] 
        
        if len(S)==l:
            print("RME_greedy did not filter")
            return np.mean(S, axis=0) 

                    
def RME_prob(params, thresh = 3):

    (G, S, indicator, m, d, eps, tau) = params

    S = NP(S, m, d, tau)
    
    while True:
        
        mu_e = np.mean(S, axis=0) 
        cov_e = np.cov(S, rowvar=0)
        M = cov_e - np.identity(d) 

        ev, v = scipy.linalg.eigh(M, eigvals=(d-1,d-1))
        v = v.reshape(len(v),)

        if ev < thresh * np.sqrt(np.log(1/eps)):
            return mu_e

        if len(S)==0: 
            print("Everything pruned")
            return 0

        Sv = (S-mu_e).dot(v)
        Z = LA.norm(Sv[np.abs(Sv) > thresh * np.sqrt(np.log(1/eps))])**2

        if np.abs(Z) < 1e-12:
            print("RME_prob cannot prune further "+str(thresh))
            return mu_e

        idx = np.nonzero((np.abs(Sv) > thresh * np.sqrt(np.log(1/eps))))[0]
        idx_neg = np.nonzero((np.abs(Sv) <= thresh * np.sqrt(np.log(1/eps))))[0]

        keep = np.nonzero(np.random.binomial(1, 1-((Sv[idx])**2/Z)))[0]
        S = np.append(S[idx_neg], S[idx[keep]], axis=0)

"""
Plotting functions now
"""

# def loss_distance_RME(params, fdr=0.1, bounds, gran):
#     (G, S, m, eps, tau) = params
#     (Low, Up) = bounds
    
#     results = {}

#     for d in range(Low,Up):
#         true_mean = 0

#         G = np.random.randn(m, d) + true_mean

#         for i in range(1, gran):

#             S = G.copy()
#             S[int(m*(1-eps)):] = true_mean + (2*i/gran)*np.sqrt(np.log(m*d/tau))
#             indicator = np.ones(len(S))
#             indicator[int(m*(1-eps)):] = 0
#             params = (G, S.copy(), indicator, m, d, eps, tau)

#             O = LA.norm(true_mean - np.mean(S[:int(m*(1-eps))], axis=0))

#             for key in ['RME']:
#                 results.setdefault(key+str(d), []).append(LA.norm(true_mean - eval(key+'(params, fdr={})'.format(fdr)))/eps)

#             results.setdefault('oracle'+str(d), []).append(O/eps)
#             results.setdefault('eps'+str(d), []).append(1)
    
#     return results


def loss_distance(params, bounds, gran):
    (G, S, m, eps, tau) = params
    (Low, Up) = bounds
    
    results = {}

    for d in range(Low,Up):
        true_mean = 0

        G = np.random.randn(m, d) + true_mean

        for i in range(1, gran):

            S = G.copy()
            S[int(m*(1-eps)):] = true_mean + (2*i/gran)*np.sqrt(np.log(m*d/tau))
            indicator = np.ones(len(S))
            indicator[int(m*(1-eps)):] = 0
            params = (G, S.copy(), indicator, m, d, eps, tau)

            O = LA.norm(true_mean - np.mean(S[:int(m*(1-eps))], axis=0))

            for key in ['NPmean','RME','RME_greedy', 'RME_prob', 'Mean']:
                results.setdefault(key+str(d), []).append(LA.norm(true_mean - eval(key+'(params)'))/eps)

            results.setdefault('oracle'+str(d), []).append(O/eps)
            results.setdefault('eps'+str(d), []).append(1)
    
    return results


def ICML2017Noise(d, m, eps,toggle=0): 

    G = np.random.randn(m, d) + np.ones(d)
    S = G.copy()
    L = int(m*(1-eps))
    Mid = int((L+m)/2)

    if toggle==0:

        " Both types of noise "

        S[L:Mid] = np.random.binomial(1, 0.5*np.ones((Mid-L)*d).reshape((Mid-L),d))

        x1 = np.append(np.array([1,0]), np.zeros(d-2))
        x2 = np.append(np.array([0,1]), np.zeros(d-2))

        b = np.random.binomial(1,0.5*np.ones(m - Mid))
        c = np.random.binomial(1,0.5*np.ones(m - Mid))
     
        b = 0*b + 12*(1-b)
        c = -2*c + 0*(1-c)
     
        S[Mid:m] = np.outer(b, x1) + np.outer(c, x2)

    if toggle==1:

        " Only hypercube noise " 

        S[L:] = np.random.binomial(1, 0.5*np.ones((m-L)*d).reshape((m-L),d))

    if toggle==2:        

        " Only second kind of noise "

        x1 = np.append(np.array([1,0]), np.zeros(d-2))
        x2 = np.append(np.array([0,1]), np.zeros(d-2))

        b = np.random.binomial(1,0.5*np.ones(m-L))
        c = np.random.binomial(1,0.5*np.ones(m-L))
     
        b = 0*b + 12*(1-b)
        c = -2*c + 0*(1-c)
     
        S[L:m] = np.outer(b, x1) + np.outer(c, x2)


    indicator = np.zeros(len(S))
    indicator[:L] = 1
    
    return (G, S, indicator)


def ICML2017Noise_sp(d, k, m, eps,toggle=0): 

    G = np.random.randn(m, d) + np.append(np.ones(k), np.zeros(d-k))
    S = G.copy()
    L = int(m*(1-eps))
    Mid = int((L+m)/2)

    if toggle==0:

        " Both types of noise "

        S[L:Mid] = np.random.binomial(1, 0.5*np.ones((Mid-L)*d).reshape((Mid-L),d))

        x1 = np.append(np.array([1,0]), np.zeros(d-2))
        x2 = np.append(np.array([0,1]), np.zeros(d-2))

        b = np.random.binomial(1,0.5*np.ones(m - Mid))
        c = np.random.binomial(1,0.5*np.ones(m - Mid))
     
        b = 0*b + 12*(1-b)
        c = -2*c + 0*(1-c)
     
        S[Mid:m] = np.outer(b, x1) + np.outer(c, x2)

    if toggle==1:

        " Only hypercube noise " 

        S[L:] = np.random.binomial(1, 0.5*np.ones((m-L)*d).reshape((m-L),d))

    if toggle==2:        

        " Only second kind of noise "

        x1 = np.append(np.array([1,0]), np.zeros(d-2))
        x2 = np.append(np.array([0,1]), np.zeros(d-2))

        b = np.random.binomial(1,0.5*np.ones(m-L))
        c = np.random.binomial(1,0.5*np.ones(m-L))
     
        b = 0*b + 12*(1-b)
        c = -2*c + 0*(1-c)
     
        S[L:m] = np.outer(b, x1) + np.outer(c, x2)

    
    indicator = np.zeros(len(S))
    indicator[:L] = 1

    return (G, S, indicator)



def loss_dimension2017(m, eps, tau, bounds, scale=1):

    (Low, Up) = bounds
    
    results = {}

    for d in scale*np.arange(Low,Up):

        G, S, indicator = ICML2017Noise(d, m, eps)
        params = (G, S, indicator, m, d, eps, tau)

        for key in ['NPmean','RME','RME_greedy', 'RME_prob', 'Mean']:
            results.setdefault(key+str(d), []).append(LA.norm(np.ones(d) - eval(key+'(params)'))/eps)

        results.setdefault('eps'+str(d), []).append(1)
    
    return results

def data_params_rmprob(filename, p1, p2, tm, params):

    L1, U1, step1 = p1
    L2, U2, step2 = p2

    f = open(filename, 'w')

    for y in np.arange(L1, U1, step1):
        for x in np.arange(L2, U2, step2):
            f.write(str((y, x, LA.norm(tm - RME_prob(params, y, x))))+'\n')

    f.close()


"SPARSE STUFF"

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

def RSPCA_matrix(params, f=0, plot=0):
    S, d, k, m, eps, tau = params
    Id = np.identity(d)
    
    S_cov = S[:,:,np.newaxis] * S[:,np.newaxis,:] - Id

    S_cov = S_cov.reshape((m, d**2))
    
    mu = np.mean(S_cov, axis=0)
    u = np.argpartition(mu, -k**2)[-k**2:]
    
    S_cov_r = S_cov[np.ix_(np.arange(m), u)]

    params2 = S_cov_r, len(u), m, eps, tau
    res = Filter2ndMoments(params2, mu[u], f, plot)
    
    if res[0]==0:
        idx = res[1]
        params = S[idx], d, k, len(S[idx]), eps, tau
        return (0, params)
    else:
        z = np.zeros(d**2)
        z[u] = mu[u]
        z = z.reshape((d,d))
        _, ans = scipy.linalg.eigh(z, eigvals=(d-1,d-1))
        ans=ans.reshape(d,)
        
        v = np.append(np.ones(k), np.zeros(d-k)); v = v/np.linalg.norm(v)
        print("2-norm diff", LA.norm(mu.reshape(d,d) - np.outer(v,v)))
        
        return (1, ans, mu)
    
    
def RSPCA_matrix2(params, f=0, plot=0, fdr = 0.2, tau = 0.2):
    S, indicator, d, k, m, eps, v_prev, thresh = params
    
    if plot==1:
        print("Total bad points: ", np.sum(1-indicator))
  
    #    """Temporary no naive prune"""

    T_naive = 2*np.sqrt(np.log(m*d/tau))
   
    med = np.median(S, axis=0)
    idx = (np.max(np.abs(med-S), axis=1) < T_naive)
    S =  S[idx]
    indicator = indicator[idx]
    if plot==1:
        print("Bad points after NP: ", np.sum(1-indicator))
    
    m = len(S)

    Id = np.identity(d)
    
#     print("building S_cov")
    
    S_cov = S[:,:,np.newaxis]  * S[:,np.newaxis,:] - Id
    S_cov = S_cov.reshape((m, d**2))
    
#     print("vectorized S_cov")
    
    mu = np.mean(S_cov, axis=0)
    u = np.argpartition(mu, -k**2)[-k**2:]

    Mask = np.zeros(d**2)
    Mask[u] = 1
    Mask = Mask.reshape(d,d)
    indices = np.nonzero(Mask)
    
    Q = np.dstack(indices)[0]
    Q2 = np.array([np.tile(Q, (len(Q),1)), np.repeat(Q, len(Q), 0)]).transpose([1,0,2])
    
    Cov = Id + np.outer(v_prev, v_prev)

    T1 = (Cov[Q2[:,0,0], Q2[:,1,1]] * Cov[Q2[:,0,1], Q2[:,1,0]]).reshape(k**2, k**2)
    vecCov = (np.identity(d) + np.outer(v_prev,v_prev)).reshape((d**2,))[u]
    T2 = np.outer(vecCov, vecCov)
    
#     print("identified restriction")

#     tS = np.random.multivariate_normal(np.zeros(d), Id + np.outer(v_prev, v_prev), (2*m, ))
#     tS_cov = tS[:,:,np.newaxis]  * tS[:,np.newaxis,:] - Id
#     tS_cov = tS_cov.reshape((2*m, d**2))
    
#     tS_cov_r = tS_cov[np.ix_(np.arange(2*m), u)]
    S_cov_r = S_cov[np.ix_(np.arange(m), u)]
    
#     vecCov = (np.identity(d) + np.outer(v_prev,v_prev)).reshape((d**2,))
#     Term2 = np.outer(vecCov, vecCov)
#     Term1 = np.kron(np.identity(d)+np.outer(v_prev,v_prev), np.identity(d) +np.outer(v_prev,v_prev))
    
    cov = np.cov(S_cov_r, rowvar=0)
    tcov = T1 + T2
    
    print("estimated cov")
    
#     M = cov - (Term2 + Term1)[np.ix_(u,u)]
    M = cov - tcov
    
    ev, v = scipy.linalg.eigh(cov, eigvals=(k**2-1,k**2-1))
    v = v.reshape(len(v),)
    print("eigenvalue:", ev)
    print("thresh:", thresh)
    print("ev < thresh:", ev < thresh)

    if ev < thresh:
        print("VALID OUTPUT")
        z = np.zeros(d**2)
        z[u] = mu[u]
        z = z.reshape((d,d))
        _, ans = scipy.linalg.eigh(z, eigvals=(d-1,d-1))
        ans=ans.reshape(d,)
#         return (1, ans)
        u2 = np.argpartition(np.abs(ans), -k)[-k:]
        z = np.zeros(d)
        z[u2] = ans[u2]
        return (1, z)

    params2 = S_cov_r, indicator, v, m, len(u), eps
    idx = filter_m_rspca(params2, f=f, plot=plot, fdr = fdr)
    
    if len(idx)==len(S):
        print("did not prune!")
        z = np.zeros(d**2)
        z[u] = mu[u]
        z = z.reshape((d,d))
        _, ans = scipy.linalg.eigh(z, eigvals=(d-1,d-1))
        ans=ans.reshape(d,)
#         return (1, ans)
        u2 = np.argpartition(np.abs(ans), -k)[-k:]
        z = np.zeros(d)
        z[u2] = ans[u2]
        return (1, z)

    params = S[idx], indicator[idx], d, k, len(S[idx]), eps, v_prev, thresh
    
    z = np.zeros(d**2)
    z[u] = mu[u]
    z = z.reshape((d,d))
    _, ans = scipy.linalg.eigh(z, eigvals=(d-1,d-1))
    ans=ans.reshape(d,)
    u2 = np.argpartition(np.abs(ans), -k)[-k:]
    z = np.zeros(d)
    z[u2] = ans[u2]
    print("Filtered")

#     print("Candidate", z)

    return (0, params)


"RSPCA_dense over here!!!!"

# def RSPCA_dense(params, f=0, plot=0, fdr = 0.5, tau = 0.2):
#     S, indicator, d, k, m, eps, v_prev, thresh = params
        
#     if plot==1:
#         print("Total bad points: ", np.sum(1-indicator))
#     T_naive = 2*np.sqrt(np.log(m*d/tau))
   
#     med = np.median(S, axis=0)
#     idx = (np.max(np.abs(med-S), axis=1) < T_naive)
#     S =  S[idx]
#     indicator = indicator[idx]
    
#     if plot==1:
#         print("Bad points after NP: ", np.sum(1-indicator))
    
#     m = len(S)

#     Id = np.identity(d)
    
#     Scov = np.cov(S, rowvar=0)
    
#     ev, v = scipy.linalg.eigh(cov, eigvals=(d**2-1,d**2-1))
    
#     Sdots = S.dot(v)
#     l = len(Sdots)
#     iqr_min = np.sort(Sdots)[int(0.25*l)]
#     iqr_max = np.sort(Sdots)[int(0.75*l)]
    
#     if ev > iqr_min and ev < iqr_max:
#         return v
#     else:
        
        
        
    
    
    
    
    
    
#     print("building S_cov")
    
#     S_cov = S[:,:,np.newaxis]  * S[:,np.newaxis,:] - Id
#     S_cov = S_cov.reshape((m, d**2))
    
#     print("vectorized S_cov")
    
#     mu = np.mean(S_cov, axis=0)
    
#     print("building estimate")
#     Mask = np.ones(d**2).reshape(d,d)
#     indices = np.nonzero(Mask)
#     Q = np.dstack(indices)[0]
#     Q2 = np.array([np.tile(Q, (len(Q),1)), np.repeat(Q, len(Q), 0)]).transpose([1,0,2])
    
#     Cov = Id + np.outer(v_prev, v_prev)

#     T1 = (Cov[Q2[:,0,0], Q2[:,1,1]] * Cov[Q2[:,0,1], Q2[:,1,0]]).reshape(d**2, d**2)
#     vecCov = (np.identity(d) + np.outer(v_prev,v_prev)).reshape((d**2,))
#     T2 = np.outer(vecCov, vecCov)


# #     tS = np.random.multivariate_normal(np.zeros(d), Id + np.outer(v_prev, v_prev), (2*m, ))
# #     tS_cov = tS[:,:,np.newaxis]  * tS[:,np.newaxis,:] - Id
# #     tS_cov = tS_cov.reshape((2*m, d**2))
    
# #     tS_cov_r = tS_cov[np.ix_(np.arange(2*m), u)]
# #     S_cov_r = S_cov[np.ix_(np.arange(m), u)]
    
# #     vecCov = (np.identity(d) + np.outer(v_prev,v_prev)).reshape((d**2,))
# #     Term2 = np.outer(vecCov, vecCov)
# #     Term1 = np.kron(np.identity(d)+np.outer(v_prev,v_prev), np.identity(d) +np.outer(v_prev,v_prev))
    
#     cov = np.cov(S_cov, rowvar=0)
#     tcov = T1 + T2
# #     tcov = np.cov(tS_cov, rowvar=0)

#     M = cov - tcov
    
#     ev, v = scipy.linalg.eigh(cov, eigvals=(d**2-1,d**2-1))
#     v = v.reshape(len(v),)

#     if ev < thresh:
#         print("VALID OUTPUT")
# #         z = np.zeros(d**2)
# #         z[u] = mu[u]
#         mu = mu.reshape((d,d))
#         _, ans = scipy.linalg.eigh(mu, eigvals=(d-1,d-1))
#         ans=ans.reshape(d,)
# #         return (1, ans)
#         u2 = np.argpartition(np.abs(ans), -k)[-k:]
#         z = np.zeros(d)
#         z[u2] = ans[u2]
#         return (1, z)

#     params2 = S_cov, indicator, v, m, d, eps
#     idx = filter_m_rspca(params2, f=f, plot=plot, fdr = fdr)
    
#     if len(idx)==len(S):
#         print("did not prune!")
# #         z = np.zeros(d**2)
# #         z[u] = mu[u]
#         mu = mu.reshape((d,d))
#         _, ans = scipy.linalg.eigh(mu, eigvals=(d-1,d-1))
#         ans=ans.reshape(d,)
# #         return (1, ans)
#         u2 = np.argpartition(np.abs(ans), -k)[-k:]
#         z = np.zeros(d)
#         z[u2] = ans[u2]
#         return (1, z)

#     params = S[idx], indicator[idx], d, k, len(S[idx]), eps, v_prev, thresh
    
# #     z = np.zeros(d**2)
# #     z[u] = mu[u]
#     mu = mu.reshape((d,d))
#     _, ans = scipy.linalg.eigh(mu, eigvals=(d-1,d-1))
#     ans=ans.reshape(d,)
#     u2 = np.argpartition(np.abs(ans), -k)[-k:]
#     z = np.zeros(d)
#     z[u2] = ans[u2]

# #     print("Candidate", z)

#     return (0, params)


def Filter2ndMoments(params, mu,  f =0, plot =0):
    S, d, m, eps, tau = params
    
    if len(S) <= 1:
        print("Filter 2nd Moments filtered too many")
        return mu
    Cov = np.cov(S, rowvar = 0)

    ev, v = scipy.linalg.eigh(Cov, eigvals=(d-1,d-1))
    v = v.reshape(len(v),)
     
    if ev > 1.1:
        
        Z = np.random.triangular(0,1,1)
        Sdots = np.abs((S-mu).dot(v))
        maxdots = np.max(Sdots)
#        print(maxdots)
        T = Z*maxdots
#        print(T)
        idx = np.nonzero((Sdots < T)) 
        
        if plot==1:
            plt.figure(f)
            plt.plot(np.arange(len(Sdots)), np.sort(Sdots), label="<x-mu,v>")
            plt.hlines(T,0, len(Sdots), label="Z*maxdots")
            plt.xlabel("sample")
            plt.ylabel("dot-product")
            plt.title("pruning for covariance estimation")

        return (0,idx)
    else:
        return (1, mu)



def dat_samp_vs_k(params2):
    klow, khigh, kstep, biter, numiter, d, k2, tau, eps = params2

    Runs1= []
    for i in range(numiter):
        ans = []
        m = 10
        for k in np.arange(klow,khigh,kstep):
            L = 100
            count = 0
            while count < 7:
                count = 0
                m+=10
                for i in range(10):            
                    v = np.append(np.ones(k), np.zeros(d-k)); v = v/np.linalg.norm(v)
                    v2 = np.append(np.zeros(d-k2), np.ones(k2)); v = v/np.linalg.norm(v)

                    cov = np.identity(d) + np.outer(v,v)
                    cov3 = np.identity(d) + np.outer(v2,v2)

                    mean = np.zeros(d) 
                    print("initialzing G now")
                    G = np.random.multivariate_normal(mean, cov, (m,))
                    G1 = G2 = np.random.multivariate_normal(mean, cov3, (m,))
#                     G1 = np.random.multivariate_normal(mean, cov3, (m, ))
                    print("initialized G")

                    S = G.copy()

                    L = int(m*(1-eps))
                    M = int((L+m)/2)
                    S[L:M] = G2[L:M]
                    S[M:m] = G1[M:m]
                    indicator = np.zeros(len(S))
                    indicator[:int(m*(1-eps))] = 1

                    params = (S.copy(), indicator, d, k, m, eps, 0, 3)
                    v2 = 0
                    f=0
                    eps_prev = eps**(1/2)
                    for i in range(biter):
#                         print("Count: ",i)
                        while v2==0:
#                             res = RSPCA_dense(params, f, plot=0, fdr=0.4)
                            res = RSPCA_matrix2(params, f, plot=0, fdr=0.2)
                            params = res[1]
                            f+=1
                        #    print(res)
                            v2 = res[0]
                        eps_prev = (eps*eps_prev)**(1/2)
                        thresh = min(2, 2*eps_prev)
            #             print("Thresh = ", thresh)
            #             print("eps_prev =", eps_prev)
                        params = (S.copy(), indicator, d, k, m, eps, res[1], thresh)
                    L = LA.norm(np.outer(res[1], res[1]) - np.outer(v,v))
                    print("k =", k, "m = ", m, "L = ", L, "eps^{1/3} = ", eps**(1/3), "count", count)
                    if L <=  eps: 
                        count +=1
            ans.append(m)
        Runs1.append(ans)

#     Runs2 = []
#     for i in range(numiter):
#         ans = []
#         m = 10
#         for k in np.arange(klow,khigh,kstep):
#             L = 100
#             count = 0
#             while count < 7:
#                 count = 0
#                 m+=10
#                 for i in range(10):            
#                     v = np.append(np.ones(k), np.zeros(d-k)); v = v/np.linalg.norm(v)
#                     v2 = np.append(np.zeros(d-k2), np.ones(k2)); v = v/np.linalg.norm(v)

#                     cov = np.identity(d) + np.outer(v,v)
# #                     cov2 = np.identity(d) - np.outer(v,v)
#                     cov3 = np.identity(d) + np.outer(v2,v2)

#                     mean = np.zeros(d) 

#                     print("initialzing G now")
#                     G = np.random.multivariate_normal(mean, cov, (m,))
#                     G2 = np.random.multivariate_normal(mean, cov3, (m,))
#                     G1 = np.random.multivariate_normal(mean, cov3, (m, ))
#                     print("initized G")

#                     S = G.copy()

#                     L = int(m*(1-eps))
#                     M = int((L+m)/2)
#                     S[L:M] = G2[L:M]
#                     S[M:m] = G1[M:m]
#                     indicator = np.zeros(len(S))
#                     indicator[:int(m*(1-eps))] = 1

#                     params = (S.copy(), indicator, d, k, m, eps, 0, 3)
#                     v2 = 0
#                     f=0
#                     eps_prev = eps**(1/2)
#                     for i in range(biter):
# #                         print("Count: ",i)
#                         while v2==0:
# #                             res = RSPCA_matrix2(params, f, plot=0, fdr=0.4)
#                             res = RSPCA_dense(params, f, plot=0, fdr=0.2)
#                             params = res[1]
#                             f+=1
#                         #    print(res)
#                             v2 = res[0]
#                         eps_prev = (eps*eps_prev)**(1/2)
#                         thresh = min(2, 2*eps_prev)
#             #             print("Thresh = ", thresh)
#             #             print("eps_prev =", eps_prev)
#                         params = (S.copy(), indicator, d, k, m, eps, res[1], thresh)
#                     L = LA.norm(np.outer(res[1], res[1]) - np.outer(v,v))
#                     print("k =", k, "m = ", m, "L = ", L, "eps^{1/3} = ", eps**(1/3), "count", count)
#                     if L <=  1.5*eps**(1/3): 
#                         count +=1
#             ans.append(m)
#         Runs2.append(ans)
#     return (Runs1, Runs2)
    return Runs1
