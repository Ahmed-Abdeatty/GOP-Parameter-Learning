import numpy as np
from optimizer import Optimizer
import queue as Q
from constraints import Constraints
import math
class Node:
    def __init__(self, parent,  theta, Mu, constraints, level, i_d, lagrangian_constant ):
        """TODO: Initialize new node at specific iteration and with primal solution.

        :parent: parent node in tree
        :iteration: iteration number
        :primal_soln: primal problem solution, a dictionary holding values of b and langrange multipliers
        :returns: nothing

        """
        self.Mu = Mu
        self.parent = parent
        self.level = level
        #self.iteration=iteration
        self.constraints = constraints
        self.childs = []
        self.theta = theta
        self.lagrangian_constant = lagrangian_constant

        self.id = i_d 
        
        #self.data=data
        """
        qualifying constraints as matrix of coefficients in constraints
        """
    # generate all the subproblems under this node
    def generate_childs(self, lagrangian, previous_theta, domain_sizes, priority_queue,counter, upper_bound):
        # previous_theta is theta used in last iteration (part of the lagrange variables)
        # is the priority queue holding all the nodes so far to add the childs to it
        # counter is the tie breaker in case of the existence of two nodes with the same value for Mu

        optimizer = Optimizer()

        # for each connected variable set it as the max qualifying constraint
        for i in range(len(previous_theta)):
            # it's qualifying constraint can be positive or negative
            lagrangian_constant = lagrangian.constant
            for sign in [-1]:
                cons = Constraints(domain_sizes)
                lagrangian_constant = lagrangian_constant + sign * len(previous_theta) * np.log(previous_theta[i])
                # add parent constraints to the child constraints
                cons.UB = np.copy(self.constraints.UB)
                cons.LB = np.copy(self.constraints.LB)
                cons.corelation = np.copy(self.constraints.corelation)
                cons.cons_list = optimizer.lagrange_ineq( lagrangian, previous_theta, i, sign, domain_sizes)
                cons.cons_list = cons.cons_list + optimizer.theta_constraints(domain_sizes)
                cons.cons_list = cons.cons_list + self.lagrange_cons[:]
                

                # define constraint over the sign of the max qualifying constraint
                if sign == 1:
                    cons.UB[i] = min(cons.UB[i], previous_theta[i])
                else:
                    cons.LB[i] = max(cons.LB[i], previous_theta[i])
                # define constraints between  max qualifying constraint and other qualifying constraint
                for j in range(len(previous_theta)):
                    if not i == j:
                        if sign == 1:
                            cons.corelation[j][i] = max(cons.corelation[j][i], (previous_theta[j]/previous_theta[i]))
                        else:
                            cons.corelation[i][j] = max(cons.corelation[i][j], (previous_theta[i]/previous_theta[j]))
                # create compact representation of LB and UB
                bounds = cons.get_bounds()
                # convert the corelation matrix into list of constraints
                cons.corelation_constraints()
                # prune if no new constraints have been added
                if  ( np.array_equal(cons.UB, self.constraints.UB) and np.array_equal(cons.LB, self.constraints.LB) and np.array_equal(cons.corelation, self.constraints.corelation)) :
                    #(Mu >= upper_bound - 0.1) or
                    pass
                else:
                    Mu, theta = optimizer.optimize(previous_theta, self.Mu, cons.cons_list, bounds, domain_sizes)
                    

                    if not ((Mu >= upper_bound - 0.1) or math.isnan(Mu) ):
                        flat_theta = theta.flatten_theta()
                        #print(math.log(previous_theta[i]) - math.log(flat_theta[i]))
                        #Mu =  lagrangian.constant + (len(flat_theta)  *(sign * (math.log(previous_theta[i]) - math.log(flat_theta[i]))))
                        #Mu = min(Mu, self.Mu)
                        node =  Node(self,  theta, Mu, cons, self.level+1, counter, lagrangian_constant)
                        node.lagrange_cons = self.lagrange_cons[:] + optimizer.lagrange_ineq( lagrangian, previous_theta, i, sign, domain_sizes)
                        priority_queue.put(((Mu, counter), node))
                        counter = counter + 1
                        self.childs.append(node)
        return counter



"""
cons = self.constraints[:]
                # generate lagrange constrain
                cons = cons + optimizer.lagrange_ineq( lagrangian, previous_theta, i, sign, domain_sizes)

                # define constraint over the sign of the max qualifying constraint
                cons = cons + optimizer.sign_ineq( previous_theta, i, sign)

                # define constraints between  max qualifying constraint and other qualifying constraint
                for j in range(len(previous_theta)):
                    if not i == j:
                        cons = cons + optimizer.relational_ineq(previous_theta, i, j, sign)
                temp_cons = cons[:]
                # Conduct optimization
                Mu, theta = optimizer.optimize(previous_theta, initial_Mu, cons, bounds, domain_sizes)
                node =  Node(self,  theta, Mu, temp_cons)
                node.order_between_sib = 2*(i+1)
                if sign == -1:
                    node.order_between_sib = 2*(i+1) -1
                #print(Mu)
                if   (Mu >= upper_bound - 0.1) or math.isnan(Mu) or np.array_equal(theta.ih, self.theta.ih) or node.order_between_sib == self.order_between_sib  :
                    #(Mu >= upper_bound - 0.1) or
                    pass
                else:
                    priority_queue.put(((Mu, counter), node))
                    counter = counter + 1
                    self.childs.append(node)
"""