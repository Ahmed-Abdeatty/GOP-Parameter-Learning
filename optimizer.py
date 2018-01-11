import numpy as np
from scipy.optimize import minimize
from variables import  Theta, Lagrangian
import math
class Optimizer:

    # class the define all of the optimization constraints, objective function, and conduct the optimization
    def __init__(self):
        pass
    
    # the objective is to minimize Mu, which is the first element in the flattened set of variables
    def objective(self, x):
        return x[0]
    
    def theta_constraints(self, domain_sizes):
    	"""
    	con = []
    	con = con + [{'type': 'eq', 'fun': lambda x: self.theta_h_constraints(x, domain_sizes) }]
    	for i in range(len(domain_sizes)-1):
    		for j in range(domain_sizes[-1]):
    			con = con + [{'type': 'eq', 'fun': lambda x: self.theta_ih_constraints(x, domain_sizes, i,j) }]
    	"""
    	con = []
    	for j in range(domain_sizes[-1]-1):
    		con = con + [{'type': 'ineq', 'fun': lambda x: self.theta_h_constraint(x, domain_sizes, j) }]
    	return con

    # return the lagrange inequality constraint
    def lagrange_ineq(self, lagrangian, flat_theta, max_index, sign, domain_sizes):
        con = {'type': 'ineq', 'fun': lambda x: self.lagrange_constraint(x, lagrangian, flat_theta, max_index, sign, domain_sizes) }
        return [con]
    
    def theta_h_constraints(self, x, domain_sizes):
        # unpack Mu and theta from the flattened set of variables
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(x[1:], domain_sizes)
        
        # add first term in the lagrangian to the inequality
        return (np.sum(theta.h) - 1)

    def theta_h_constraint(self, x, domain_sizes,i):
        # unpack Mu and theta from the flattened set of variables
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(x[1:], domain_sizes)
        
        # add first term in the lagrangian to the inequality
        return theta.h[i] - theta.h[i+1] - .05

    def theta_ih_constraints(self, x, domain_sizes, i, j):
        # unpack Mu and theta from the flattened set of variables
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(x[1:], domain_sizes)
        
        # for every h, i: sum_ih holds the (sum over all xi (theta(xi,xh)) - 1)
        sum_ih = np.array(([l.sum(axis=0) - 1 for l in theta.ih]))[i][j]
        return sum_ih.sum()
    # Define the lagrange inequality constraint
    def lagrange_constraint(self, x, lagrangian, flat_theta, max_index,sign, domain_sizes):
        # unpack Mu and theta from the flattened set of variables
        Mu = x[0]
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(x[1:], domain_sizes)
        # add Mu to the inequality
        ineq = Mu
        # add first term in the lagrangian to the inequality
        ineq = ineq - (lagrangian.coefficient_theta_h * (np.sum(theta.h) - 1))
        # add second term in the lagrangian to the inequality
        # for every h, i: sum_ih holds the (sum over all xi (theta(xi,xh)) - 1)
        sum_ih = np.array(([l.sum(axis=0) - 1 for l in theta.ih]))
        ineq = ineq - (np.multiply(lagrangian.coefficients_theta_ih,sum_ih).sum())
        # add the constant term in the lagrangian to the inequality
        ineq = ineq - lagrangian.constant
        # finally add the max qualifying constraint
        # len(flat_theta) is the number of connected variables
        ineq = ineq + (len(flat_theta) * (sign * (math.log(flat_theta[max_index]) - math.log(x[max_index+1]))))

        return ineq

    # define constraints between  max qualifying constraint and other qualifying constraint
    def relational_ineq(self, flat_theta, max_index, min_index, sign):
        # max_index is the index of the max qualifying constraint
        # min_index is the index of one of the non max qualifying constraint
        # sign is an indecator (takes values 1 or -1) to indecate the sign 
        con1 = {'type': 'ineq', 'fun': lambda x: sign * (flat_theta[max_index] * x[min_index+1] - flat_theta[min_index] * x[max_index+1])}
        con2 = {'type': 'ineq', 'fun': lambda x: sign * (flat_theta[max_index] * flat_theta[min_index] - x[min_index+1] * x[max_index+1])}
        return [con1, con2]

    # define constraint over the sign of the max qualifying constraint
    def sign_ineq(self, flat_theta, max_index, sign):
        con = {'type': 'ineq', 'fun': lambda x: sign * (flat_theta[max_index] - x[max_index+1])}
        return [con]

    # conduct the optimization
    # returns Mu, optimal theta
    def optimize(self, initial_theta, initial_Mu, constraints, bounds, domain_sizes):
        initial_theta = np.insert(initial_theta,0,initial_Mu)
        sol = minimize(self.objective, initial_theta, method= 'SLSQP',options = {'ftol':1e-9}, bounds= bounds, constraints = constraints)
        # unpack Mu and theta from the flattened solution
        Mu = sol.x[0]
        if sol.success == False:
        	return float('NaN'), float('NaN')
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(sol.x[1:], domain_sizes)
        

        return Mu, theta


    
