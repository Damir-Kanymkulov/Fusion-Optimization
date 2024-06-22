import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy as np
import copy    # array-copying convenience
import sys     # max float
import numpy.random as rnd
from early_fusion import MLP
from intermediate_fusion_brute_force_search import NewMyEnsemble, new_train_intermediate

#-------------------------------------------Definition of functions---------------------------------------------------------
# Define the PSO algorithm
#particle class
class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        # initialize position of the particle with 0.0 value
        self.position = [0.0 for i in range(dim)]
        # initialize velocity of the particle with 0.0 value
        self.velocity = [0.0 for i in range(dim)]
        # initialize best particle position of the particle with 0.0 value
        self.best_part_pos = [0.0 for i in range(dim)]
        # loop dim times to calculate random position and velocity
        # range of position and velocity is [minx, max]
        for i in range(dim):
            self.position[i] = ((maxx - minx) *
                self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) *
                self.rnd.random() + minx)

        # compute fitness of particle
        self.fitness = fitness(self.position) # curr fitness
        # initialize best position and fitness of this particle
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness # best fitness


# particle swarm optimization function
def PSO(fitness, max_iter, n, dim, minx, maxx):
    # hyper parameters
    w = 0.729    # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)

    rnd = random.Random(0)

    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

    # compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max # swarm best

    # computer best particle of swarm and it's fitness
    for i in range(n): # check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)

    # main loop of pso
    Iter = 0
    while Iter < max_iter:
        # after every 10 iterations
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)

        for i in range(n): # process each particle
            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random()    # randomizations
                r2 = rnd.random()

                swarm[i].velocity[k] = (
                                        (w * swarm[i].velocity[k]) +
                                        (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
                                        (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                                      )

                # if velocity[k] is not in [minx, max]
                # then clip it
                if swarm[i].velocity[k] < minx:
                    swarm[i].velocity[k] = minx
                elif swarm[i].velocity[k] > maxx:
                    swarm[i].velocity[k] = maxx

            # compute new position using new velocity
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                if swarm[i].position[k] < minx:
                    swarm[i].position[k] = minx
                elif swarm[i].position[k] > maxx:
                    swarm[i].position[k] = maxx
                # print("Swarm positions: ", swarm[i].position[k])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # compute fitness of new position
            swarm[i].fitness = fitness(swarm[i].position)

            # is new position a new best for the particle?
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)

        # for-each particle
        Iter += 1
    #end_while
    return best_swarm_pos, best_swarm_fitnessVal
  # end pso


def fitness_function_factory_PSO(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
    def calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion):
        train_loaders, val_loaders = [], []
        models = []
        index = 0
        for data_type in loaders_dict.keys():
            train_loaders_per_type, val_loaders_per_type = [], []
            for i, (train_loader, val_loader) in enumerate(loaders_dict[data_type]):
                train_loaders_per_type.append(train_loader)
                val_loaders_per_type.append(val_loader)
                input_dim = dimension_dict[data_type]
                model = MLP(input_dim=input_dim, n_layers=round(solution[index]), fusion="intermediate")
                models.append(model)
                index += 1
            train_loaders.extend(train_loaders_per_type)
            val_loaders.extend(val_loaders_per_type)
        model = NewMyEnsemble(models, n_layers=round(solution[index]))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #Training
        model_path = 'PSO_best_model_min_val_loss.pth'
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']
        return loss

    def fitness_func_PSO(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        return solution_fitness
    return fitness_func_PSO

#------------------------------------------------- PSO optimization----------------------------------------------------------
def intermediate_fusion_PSO(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, num_particles, criterion):
    fitness_func_PSO = fitness_function_factory_PSO(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    dim = sum(1 for data_type in loaders_dict.keys() for i in loaders_dict[data_type]) + 1         # Number of solutions
    lb, ub = 1, 10 # Lower and upper bounds
    w, c1, c2 = 0.5, 1, 2 #inertia weight, personal acceleration factor, social acceleration factor
    solution, fitness = PSO(fitness_func_PSO, max_iter, num_particles, dim, lb, ub)
    return np.around(solution).astype(int), fitness