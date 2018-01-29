import numpy as np
from scipy.optimize import minimize
from variables import  Theta, Lagrangian
import math
class Optimizer:

    # class the define all of the optimization constraints, objective function, and conduct the optimization
    def __init__(self):
        pass
    
    def corelation_constraint(self, max_index, min_index, constant):
        return [{'type': 'ineq', 'fun': lambda x: x[max_index+1] - constant * x[min_index+1]} ]

    # the objective is to minimize Mu, which is the first element in the flattened set of variables
    def objective(self, x):
        return x[0]
    
    # add the sum to one constraint + the ordering of theta_h
    def theta_constraints(self, domain_sizes):
        
        con = []
        con = con + [{'type': 'eq', 'fun': lambda x: self.theta_h_constraints(x, domain_sizes) }]
        for i in range(len(domain_sizes)-1):
            for j in range(domain_sizes[-1]):
                con = con + [{'type': 'eq', 'fun': lambda x: self.theta_ih_constraints(x, domain_sizes, i,j) }]
        
        for j in range(domain_sizes[-1]-1):
            con = con + [{'type': 'ineq', 'fun': lambda x: self.theta_h_constraint(x, domain_sizes, j) }]
        return con

    # return the lagrange inequality constraint
    def lagrange_ineq(self, lagrangian, flat_theta, max_index, sign, domain_sizes):
        con1 = {'type': 'ineq', 'fun': lambda x: self.lagrange_constraint(x, lagrangian, flat_theta, max_index, sign, domain_sizes) }
        #con2 = {'type': 'ineq', 'fun': lambda x: self.Mu_constraint(x, lagrangian, flat_theta, max_index, sign, domain_sizes)
        #,'jac': lambda x: self.Mu_jac(x, lagrangian, flat_theta, max_index, sign, domain_sizes) }
 
        return [con1]
    
    def positive_Mu(self):
        return [{'type': 'ineq', 'fun': lambda x: x[0] }]
    # theta_h sum to one
    def theta_h_constraints(self, x, domain_sizes):
        # unpack Mu and theta from the flattened set of variables
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(x[1:], domain_sizes)
        
        # add first term in the lagrangian to the inequality
        return (np.sum(theta.h) - 1)

    # theta_h ordering
    def theta_h_constraint(self, x, domain_sizes,i):
        # unpack Mu and theta from the flattened set of variables
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(x[1:], domain_sizes)
        
        # add first term in the lagrangian to the inequality
        return theta.h[i] - theta.h[i+1] - .05

    # theta_ih sum to one
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
        #ineq = ineq - (lagrangian.coefficient_theta_h * (np.sum(theta.h) - 1))
        # add second term in the lagrangian to the inequality
        # for every h, i: sum_ih holds the (sum over all xi (theta(xi,xh)) - 1)
        #sum_ih = np.array(([l.sum(axis=0) - 1 for l in theta.ih]))
        #ineq = ineq - (np.multiply(lagrangian.coefficients_theta_ih,sum_ih).sum())
        # add the constant term in the lagrangian to the inequality
        ineq = ineq - lagrangian.constant
        # finally add the max qualifying constraint
        # len(flat_theta) is the number of connected variables * weights[max_index]
        if sign == -1:
            ineq = ineq + (len(flat_theta)  *(sign * (math.log(flat_theta[max_index]) - math.log(x[max_index+1]))))

        return ineq

    # Define the lagrange inequality constraint (forcing Mu >= 0)
    def Mu_constraint(self, x, lagrangian, flat_theta, max_index,sign, domain_sizes):
        # unpack Mu and theta from the flattened set of variables
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(x[1:], domain_sizes)
        
        # add the constant term in the lagrangian to the inequality
        ineq =  lagrangian.constant
        # finally add the max qualifying constraint
        # len(flat_theta) is the number of connected variables
        if sign == -1:
            ineq = ineq - (len(flat_theta) * (sign * (math.log(flat_theta[max_index]) - math.log(x[max_index+1]))))

        return ineq 
    def lagrange_jac(self, x, lagrangian, flat_theta, max_index,sign, domain_sizes):
        # unpack Mu and theta from the flattened set of variables
        jac = np.zeros(len(flat_theta)+1)
        jac[0] = 1
        if sign == -1:
            #* weights[max_index] 
            jac[max_index+1] = -1 * len(flat_theta) * sign / x[max_index+1]
        return jac

    def Mu_jac(self, x, lagrangian, flat_theta, max_index,sign, domain_sizes):
        # unpack Mu and theta from the flattened set of variables
        jac = np.zeros(len(flat_theta)+1)
        if sign == -1:
            jac[max_index+1] = -1 * len(flat_theta) * sign / x[max_index+1]
        return jac
    # define constraints between  max qualifying constraint and other qualifying constraint
    def relational_ineq(self, flat_theta, max_index, min_index, sign):
        # max_index is the index of the max qualifying constraint
        # min_index is the index of one of the non max qualifying constraint
        # sign is an indecator (takes values 1 or -1) to indecate the sign 
        con1 = {'type': 'ineq', 'fun': lambda x: sign * (flat_theta[max_index] * x[min_index+1] - flat_theta[min_index] * x[max_index+1] )}
        #con2 = {'type': 'ineq', 'fun': lambda x: sign * (flat_theta[max_index] * flat_theta[min_index] - x[min_index+1] * x[max_index+1]-0.005)}
        return [con1]

    # define constraint over the sign of the max qualifying constraint
    def sign_ineq(self, flat_theta, max_index, sign):
        con = {'type': 'ineq', 'fun': lambda x: sign * (flat_theta[max_index] - x[max_index+1])}
        return [con]

    # conduct the optimization
    # returns Mu, optimal theta
    def optimize(self, initial_theta, initial_Mu, constraints, bounds, domain_sizes):
        initial_theta = np.insert(initial_theta,0,initial_Mu)
        sol = minimize(self.objective, initial_theta, method= 'SLSQP',tol="1e-6", bounds= bounds, constraints = constraints)
        # unpack Mu and theta from the flattened solution ,tol="1e-20"
        #print(sol)
        Mu = sol.x[0]
        print(sol.message)
        if sol.success == False:
            return float('NaN'), float('NaN')
        theta = Theta(domain_sizes)
        theta = theta.array_to_theta(sol.x[1:], domain_sizes)
        

        return Mu, theta


    
