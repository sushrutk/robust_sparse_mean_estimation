#!/usr/bin/env python
from scipy import array,real,dot,column_stack,row_stack,append
import numpy
import time
ra = numpy.random
la = numpy.linalg
import numpy as np
import scipy
import copy
from scipy import special
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pylab import rcParams
import pickle
from matplotlib import rc
import ast

def err_rspca(a,b): return LA.norm(np.outer(a,a)-np.outer(b, b))

def err(a,b): return LA.norm(a-b)

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
    def __init__(self, d = 0, m = 0, eps = 0, k = 0, tau = 0.2, mass = 0, tv = 0, fv = 0):
        self.d = d
        self.m = m
        self.eps = eps
        self.k = k
        self.tau = tau
        self.mass = mass
        self.tv = tv
        self.fv = fv


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


class MixtureMean(object):
    def __init__(self):
        pass

    def generate(self, params):
        d, k, eps, m, tau = params.d, params.k, params.eps, params.m, params.tau

        # tm = np.append(np.ones(k), np.zeros(d-k))
        tm = np.zeros(d)
        fm = np.append(np.zeros(d-k), np.ones(k))

        cov = 2*np.identity(d) - np.diag(fm)

        G = np.random.randn(m, d) + tm
        GF = (1/2)*np.random.randn(m, d) + (1/np.sqrt(2))

        S = G.copy()

        L = int(m*(1-eps))

        S[L:] = GF[L:]

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau)
        return params, S, indicator, tm


class BimodalModel(object):
    def __init__(self):
        pass
    
    def generate(self, params):
        d, k, eps, m, tau = params.d, params.k, params.eps, params.m, params.tau

        tm = np.zeros(d)
        fm = np.append(np.zeros(d-k), np.ones(k))

        cov = 2*np.identity(d) - np.diag(fm)

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

class TailFlipModel(object):
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

class RSPCA_MixtureModel(object):
    def __init__(self):
        pass

    def generate(self, params):
        d, eps, m, tau, mass, k = params.d, params.eps, params.m, params.tau, params.mass, params.k
        tv, fv = params.tv, params.fv
        # tm = np.append(np.ones(k), np.zeros(d-k))

        tcov = np.identity(d) + np.outer(tv, tv)
        fcov = np.identity(d) + mass*np.outer(fv, fv)

        G1 = np.random.multivariate_normal(np.zeros(d), tcov, (m, ))
        G2 = np.random.multivariate_normal(np.zeros(d), fcov, (m, ))
        S = G1.copy()

        L = int(m*(1-eps))
        S[L:] = G2[L:]

        indicator = np.ones(len(S))
        indicator[L:] = 0
        params = Params(d,m,eps,k,tau, mass = mass)
        return params, S, indicator, tv


def PathSPCA(A,k):
    M,N=A.shape
    # Loop through variables

    As=((A*A).sum(axis=0));
    vmax=As.max();
    vp=As.argmax();
    subset=[vp];

    vars=[];res=subset;rhos=[(A[:,vp]*A[:,vp]).sum()];
    Stemp=array([rhos])

    for i in range(1,k):

        lev,v=la.eig(Stemp)
        vars.append(real(lev).max())
        vp=real(lev).argmax()
        x=dot(A[:,subset],v[:,vp])
        x=x/la.norm(x)
        seto=range(0,N)

        for j in subset:
            seto.remove(j)

        vals=dot(x.T,A[:,seto]);vals=vals*vals
        rhos.append(vals.max())
        vpo=seto[vals.argmax()]
        Stemp=column_stack((Stemp,dot(A[:,subset].T,A[:,vpo])))
        vbuf=append(dot(A[:,vpo].T,A[:,subset]),array([(A[:,vpo]*A[:,vpo]).sum()]))
        Stemp=row_stack((Stemp,vbuf))
        subset.append(vpo)

    lev,v=la.eig(Stemp)
    vars.append(real(lev).max())

    return vars,res,rhos

# **** Run quick demo ****
# Simple data matrix with N=7 variables and M=3 samples
k = 100

d_rsm_m_1, eps_rsm_m_1 = 1000, 0.1
params_rsm_m_1 = Params(d=d_rsm_m_1, eps=eps_rsm_m_1, m = 500, k = 100)
trials_rsm_m_1 = 1

model_rsm_m_1 = DenseNoiseModel(2)

inp, S, indicator, tm = model_rsm_m_1.generate(params_rsm_m_1)

# Call function
start_time = time.time()


vars,res,rhos = PathSPCA(S,k)
print res
print vars
print rhos
print "--- %s seconds ---" % (time.time() - start_time)

