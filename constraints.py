
import numpy as np
from optimizer import Optimizer
class Constraints:
    def __init__(self,domain_sizes):
        self.no_connected_comp = domain_sizes[-1]*(np.sum(domain_sizes[:-1])+1)
         
        self.LB=np.zeros(self.no_connected_comp) + .0001
        self.UB=np.ones(self.no_connected_comp)
        self.corelation=np.zeros((self.no_connected_comp,self.no_connected_comp))
        self.cons_list = []

    def get_bounds(self):
        
        temp1 = self.LB
        temp1 = np.insert(temp1,0,-1333330.0)
        temp2 = self.UB
        temp2 = np.insert(temp2,0,250)
        temp = np.array((temp1,temp2)).T
        
        return temp
    # converts self.corelation into a list of constraints
    def corelation_constraints(self):
        optimizer = Optimizer()
        for i in range(self.no_connected_comp):
            for j in range(self.no_connected_comp):
                if not self.corelation[i][j] == 0:
                    self.cons_list = self.cons_list + optimizer.corelation_constraint(i,j,self.corelation[i][j])


    
