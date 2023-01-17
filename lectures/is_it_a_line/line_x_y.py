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
        self.sigma_x = 1.0
        self.sigma_y = 1.0
        self.order=order+1
        self.names=['{0}'.format(i) for i in range(self.order)]
        self.bounds=[[-10,10] for _ in range(self.order)]
        # add the unobserved x data points
        for i in range(data['x'].shape[0]):
            self.names.append('x_{}'.format(i))
            self.bounds.append([data['x'][i]-self.sigma_x,data['x'][i]+self.sigma_x])

    def log_likelihood(self,p):
        model = np.array([poly( p['x_{}'.format(i)], p, order=self.order) for i in range(data['x'].shape[0])])
        logL_y = -0.5*np.sum(((self.data['y']-model)/self.sigma_y)**2)
        logL_x = 0.0
        for i in range(data['x'].shape[0]):
            logL_x += -0.5*((self.data['x'][i]-p['x_{}'.format(i)])/self.sigma_x)**2
        return logL_x+logL_y
    
    def log_prior(self,p):
        logP = super(PolynominalModel,self).log_prior(p)
        return logP

if __name__=='__main__':
    # hard coded options
    out_folder = 'linear'
    order      = 1
    
    data = np.genfromtxt('data.txt',names=True)
    M=PolynominalModel(data, order = order)
    
    work=cpnest.CPNest(M, verbose=2,
                       nnest=1, nensemble=3, nlive=100, maxmcmc=5000, nslice=0, nhamiltonian=0, seed = 1,
                       resume=1, periodic_checkpoint_interval=600,output=out_folder)
    work.run()
    print("estimated logZ = {0} \pm {1}".format(work.logZ,work.logZ_error))
    
    samples = work.posterior_samples
    models = []
    for s in samples:
        models.append(np.array([poly( s['x_{}'.format(i)], s, order=M.order) for i in range(data['x'].shape[0])]))
    
    l,m,h = np.percentile(models, [5,50,95], axis=0)
    
    import matplotlib.pyplot as plt
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(data['x'],m,'-k')
    for ms in models:
        ax.plot(data['x'],ms,'-k',linewidth=0.25)
        
    ax.errorbar(data['x'],data['y'],xerr=M.sigma_x,yerr=M.sigma_y)
    ax.fill_between(data['x'],l,h,facecolor='turquoise')
    
    plt.show()


