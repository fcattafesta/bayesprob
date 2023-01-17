import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import logsumexp
import matplotlib.pyplot as plt

def loglikelihood(x,y,p,sigma=1):
    m = Polynomial(p)
    return -0.5*np.sum(((y-m(x))/sigma)**2)

def logprior(p):
    return 0.0

def logposterior(x,y,p,sigma=1):
    return logprior(p) + loglikelihood(x,y,p,sigma=sigma)

def FindHeightForLevel(inLogArr, adLevels, logdd):
    """
    Given a probability array, computes the heights corresponding to some given credible levels.
    
    Arguments:
        :np.ndarray inLogArr: probability array
        :iterable adLevels:   credible levels
        :double logdd:        variables log differential (∑ log(dx_i))
        
    Returns:
        :np.ndarray: heights corresponding to adLevels
    """
    # flatten and create reversed sorted list
    adSorted = np.ascontiguousarray(np.sort(inLogArr.flatten())[::-1])
    # create a normalized cumulative distribution
    adCum = log_cumulative(adSorted + logdd)
    # find values closest to levels
    adHeights = []
    adLevels = np.ravel([adLevels])
    for level in adLevels:
        idx = (np.abs(adCum-np.log(level))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)
    return adHeights

def log_cumulative(inarr):
    h = np.zeros(inarr.shape[0])
    h[0] = inarr[0]
    for i in range(1,inarr.shape[0]):
        h[i] = np.logaddexp(h[i-1],inarr[i])
    return h

if __name__ == "__main__":
    data = np.genfromtxt('data.txt',names=True)
    err  = np.ones(data.shape[0])
    """
    1) y = a+bx
    2) y = a+bx+cx^2
    """
    N = 100
    a = np.linspace(-10,10,N)
    b = np.linspace(-5,5,N)
    c = np.linspace(-10,10,N)
    
    A,B = np.meshgrid(a,b)
    logP_2d = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            logP_2d[i,j] = logposterior(data['x'],data['y'],(a[i],b[j]),sigma=1)
    
    levels = np.sort(FindHeightForLevel(logP_2d, [0.9], 0.0))
    
    # joint pdf
    fig = plt.figure(1)
    ax  = fig.add_subplot(221)
    C   = ax.contourf(A,B,logP_2d.T,100)
    ax.contour(A,B,logP_2d.T,levels)
    CB  = plt.colorbar(C)
    CB.set_label('$\log p(\lambda|x,y,M)$')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    # marginals
    ax  = fig.add_subplot(222)
    marginal = logsumexp(logP_2d,axis = 0)+np.log(np.diff(a)[0])
    p = np.exp(marginal)
    ax.plot(b,p)
    mb = np.sum(b*p)/np.sum(p)
    sig_b = np.sum(((b-mb)**2)*p)/np.sum(p)
    ax.set_xlabel('b')
    ax  = fig.add_subplot(223)
    marginal = logsumexp(logP_2d,axis = 1)+np.log(np.diff(b)[0])
    p = np.exp(marginal)
    ma = np.sum(a*p)/np.sum(p)
    sig_a = np.sum(((a-ma)**2)*p)/np.sum(p)
    ax.plot(a,np.exp(marginal))
    ax.set_xlabel('a')
    print(ma,mb,sig_a,sig_b)
    ax  = fig.add_subplot(224)
    ax.errorbar(data['x'],data['y'],yerr = err)
    ax.plot(data['x'],Polynomial((ma,mb))(data['x']),'-k')
    ax.fill_between(data['x'],Polynomial((ma-sig_a,mb-sig_b))(data['x']),Polynomial((ma+sig_a,mb+sig_b))(data['x']),facecolor='turquoise')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    print("log Evidence = {}".format(logsumexp(logP_2d)+np.log(np.diff(b)[0])++np.log(np.diff(a)[0])))
    plt.show()

    exit()
    
    for i in range(N):
        logP[i] = loglikelihood(data['x'],data['y'],(c[i],b[i],a[i]),sigma=1,order=2)
    
    idx = np.argsort(logP)[::-1]
    jdx = idx[:10]
    
    
    plt.errorbar(data['x'],data['y'],yerr = err)
    for i in jdx:
        print('indeces',i)
        plt.plot(data['x'],np.array([poly(xi,(c[i],b[i],a[i]),order=2) for xi in data['x']]))
    plt.show()
    exit()
        

    
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    S = ax.scatter(a,b,c = np.exp(logP))
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    plt.colorbar(S)
    plt.show()
    """
    
    plt.show()
    """
