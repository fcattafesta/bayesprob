import numpy as np
"""
to install cpnest, make sure to clone the repository from github:
git clone git@github.com:johnveitch/cpnest.git
cd cpnest
git checkout massively_parallel
git pull
python setup.py install
"""
import cpnest.model

def poly(x, p, order = 1):
    p = np.sum(np.array([p['{}'.format(i)]*x**i for i in range(order)]))
    return p

class PolynominalModel(cpnest.model.Model):
    """
    An n-dimensional gaussian
    """
    def __init__(self,data,order=1):
        self.data = data
        self.sigma = 3.0
        self.order=order+1
        self.names=['{0}'.format(i) for i in range(self.order)]
        self.bounds=[[-10,10] for _ in range(self.order)]

    def log_likelihood(self,p):
        model = np.array([poly(xi, p, order=self.order) for xi in self.data['x']])
        return -0.5*np.sum(((self.data['y']-model)/self.sigma)**2)
    
    def log_prior(self,p):
        logP = super(PolynominalModel,self).log_prior(p)
        return logP

if __name__=='__main__':
    # hard coded options
    out_folder = 'parabola'
    order      = 2
    
    data = np.genfromtxt('data.txt',names=True)
    model=PolynominalModel(data, order = order)
    
    work=cpnest.CPNest(model, verbose=2,
                       nnest=1, nensemble=1, nlive=1000, maxmcmc=5000, nslice=0, nhamiltonian=0, seed = 1,
                       resume=1, periodic_checkpoint_interval=600,output=out_folder)
    work.run()
    print("estimated logZ = {0} \pm {1}".format(work.logZ,work.logZ_error))

