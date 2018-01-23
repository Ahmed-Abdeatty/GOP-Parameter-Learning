import numpy as np
class Theta:

    """Docstring for theta. """

    def __init__(self,domain_sizes,initialize=1):
        """TODO: to be defined1. """
        domain_size_h=domain_sizes[-1]
        self.ih=[]
        for d in domain_sizes[:-1]:
            self.ih.append(np.random.rand(d,domain_size_h))
        self.h=np.ones(domain_size_h)/domain_size_h
        self.h=np.random.rand(domain_size_h)/domain_size_h
        self.h = self.h / sum(self.h)
        x = np.array(([(1.0/l.sum(axis=0)) for l in self.ih]))
        for i in range(len(x)):
            self.ih[i] = self.ih[i] * x[i] 
        if initialize==0:
            self.ih=[]
            for d in domain_sizes[:-1]:
                self.ih.append(np.zeros((d,domain_size_h))/(d))
            self.h=np.zeros(domain_size_h)/domain_size_h

    # conver theta into flat array (optimizer preference)
    def flatten_theta(self):
        return np.concatenate((self.h,np.array(self.ih).reshape(-1)))

    # convert flat array to Theta objexct
    def array_to_theta(self, flat_theta ,domain_sizes):
        domain_size_h=domain_sizes[-1]
        self.h= flat_theta[0:domain_size_h]
        self.ih=[]
        start_index = domain_size_h
        for d in domain_sizes[:-1]:
            self.ih.append(np.array(flat_theta[start_index:start_index + d*domain_size_h]).reshape(d,domain_size_h))
            start_index = start_index + d*domain_size_h
        return self

class Lagrangian:
    def __init__(self,domain_sizes):
        self.constant=0
        self.coefficient_theta_h=1
        self.coefficients_theta_ih=np.zeros((len(domain_sizes)-1,domain_sizes[-1]))



