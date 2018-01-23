# -*- coding: utf-8 -*-
from node import Node
from variables import Theta, Lagrangian
from optimizer import Optimizer
import numpy as np
from sklearn.preprocessing import normalize
import queue as Q
class Manager:

    def __init__(self,data,domain_sizes):
        self.data=data
        self.k=0
        self.q=Q.PriorityQueue()
        optimizer = Optimizer()
        self.upper_bound = 1000
        self.lower_bound = -10
        self.tol = 0.01
        #optimizer.theta_constraints(domain_sizes)
        self.counter=0
        self.domain_sizes=domain_sizes
        self.M=np.shape(data)[0]
        #optimizer.theta_constraints(self.domain_sizes)
        node = Node(None,Theta(domain_sizes),-100,[])
        node.order_between_sib = 0
        self.q.put(((0,0),node))

    def run_gop(self):
        while abs((self.upper_bound-self.lower_bound)/self.upper_bound) > self.tol:
            if self.q.empty():
                break
            else:
                self.next_iteration()
            
 
    def next_iteration(self):
        if self.k==0:
            self.theta=Theta(self.domain_sizes)
        
        
        print(self.theta.ih)
        print(self.theta.h)
        lagrangian, primal_value = self.solve_primal()
        self.upper_bound = min(self.upper_bound, primal_value)
        if self.k==0:
            self.upper_bound = primal_value
        self.k=self.k+1
        print(self.upper_bound)
        current_node=self.q.get()[1]
        print("MU")
        print(current_node.Mu)
        self.theta=current_node.theta
        self.lower_bound= min(self.lower_bound, current_node.Mu)
        #print( (current_node.Mu >= self.upper_bound - 0.1))
        flat_theta=self.theta.flatten_theta()
        b = (0.0000001, .99)
        bounds = [(-1333330.0,25)] + ([b] * len(flat_theta)  )
        self.counter=current_node.generate_childs(lagrangian,flat_theta,self.domain_sizes,bounds,current_node.Mu,self.q,self.counter,self.upper_bound)
        

    def solve_primal(self):

        lambda_h=np.log(self.theta.h)
        lambda_ih=np.log(self.theta.ih)

        # calculate b(x_h) for every h and every training case
        b_h=np.zeros((self.M,self.domain_sizes[-1]))
        b_h[:,:]=np.copy(lambda_h)
        for cnt, l in enumerate(lambda_ih):
            b_h[:,:]=np.add(b_h[:,:],l[data[:,cnt],:])
        b_h[:,:]=np.exp((1.0/self.M)*b_h)
        
        b_h=normalize(b_h,norm='l1')
        #Calculate b_bar
        b_bar=Theta(self.domain_sizes,initialize=0)
        b_bar.h=(1.0/self.M)*np.sum(b_h,axis=0)
        for i,d_i in enumerate(self.domain_sizes[:-1]):
            for xi in range(d_i):
                for m,sample in enumerate(self.data):
                    if sample[i]==xi:
                        b_bar.ih[i][xi,:]=b_bar.ih[i][xi,:]+b_h[m,:]
        # calculate lagrange multipliers
        lagrangian=Lagrangian(self.domain_sizes)

        # lagrangian.constant is the constant part of the dual (used for calculating both the primal and dual vaues)
        lagrangian.constant=-np.sum(np.multiply(np.log(self.theta.h),b_bar.h))+(1.0/self.M)*np.sum(np.multiply(b_h,np.log(b_h)))
        for i in range(len(self.domain_sizes[:-1])):
            lagrangian.constant=lagrangian.constant-np.sum(np.multiply(np.log(self.theta.ih[i]),b_bar.ih[i]))
        
        # lagrangian.coefficients_theta_ih = sum(b_bar.ih(i,h))/sum(theta.ih(i,h))
        d = np.array(([l.sum(axis=0)  for l in self.theta.ih]))
        n = np.array(([l.sum(axis=0)  for l in b_bar.ih]))
        lagrangian.coefficients_theta_ih = n/d
        lagrangian.coefficient_theta_h= np.sum( b_bar.h)/ np.sum( self.theta.h)
        

        # compute primal value
        primal_value = lagrangian.constant + (lagrangian.coefficient_theta_h * (np.sum(self.theta.h) - 1))
        sum_ih = np.array(([l.sum(axis=0) - 1 for l in self.theta.ih]))
        primal_value = primal_value + (np.multiply(lagrangian.coefficients_theta_ih,sum_ih).sum())
        return lagrangian, primal_value

data=np.asarray([[1,1,1],[1,0,0],[0,0,0],[0,1,0]])
domain_sizes=[2,2,2,2]
manager=Manager(data,domain_sizes)
manager.run_gop()
