import numpy as np
from optimizer import Optimizer
import queue as Q
import math
class Node:
    def __init__(self, parent,  theta, Mu, constraints):
        """TODO: Initialize new node at specific iteration and with primal solution.

        :parent: parent node in tree
        :iteration: iteration number
        :primal_soln: primal problem solution, a dictionary holding values of b and langrange multipliers
        :returns: nothing

        """
        self.Mu = Mu
        self.parent = parent
        #self.iteration=iteration
        self.constraints = constraints
        self.childs = []
        self.theta = theta
        
        #self.data=data
        """
        qualifying constraints as matrix of coefficients in constraints
        """
    # generate all the subproblems under this node
    def generate_childs(self, lagrangian, previous_theta, domain_sizes, bounds, initial_Mu, priority_queue,counter, upper_bound):
        # previous_theta is theta used in last iteration (part of the lagrange variables)
        # is the priority queue holding all the nodes so far to add the childs to it
        # counter is the tie breaker in case of the existence of two nodes with the same value for Mu

        optimizer = Optimizer()

        # for each connected variable set it as the max qualifying constraint
        for i in range(len(previous_theta)):
            # it's qualifying constraint can be positive or negative
            for sign in [-1]:
                # add parent constraints to the child constraints
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
                if     node.order_between_sib == self.order_between_sib or  math.isnan(Mu)  :
                    #(Mu >= upper_bound - 0.1) or
                    pass
                else:
                    priority_queue.put(((Mu, counter), node))
                    counter = counter + 1
                    self.childs.append(node)
        return counter



