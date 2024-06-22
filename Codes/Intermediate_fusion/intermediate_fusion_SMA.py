import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy as np
from early_fusion import MLP
from intermediate_fusion_brute_force_search import NewMyEnsemble, new_train_intermediate

#-------------------------------------------Definition of functions---------------------------------------------------------
def SMA(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # Initialize the positions of slime mould
    positions = np.random.uniform(lb, ub, size=(SearchAgents_no, dim))
    Best_pos = None
    Best_score = float("inf")
    w = 0.9  #  Weight parameter
    vb = 0   #  Vibration parameter
    z  = 0.1 #  Random parameter
    for i in range(Max_iter):
        # Calculate the fitness value for each agent
        fitness_value = np.array([objf(pos) for pos in positions])

        # Sort the agents based on fitness value from best to worst
        sorted_indices = np.argsort(fitness_value)
        positions = positions[sorted_indices]

        # Update the weights
        w = w * np.exp(-i / Max_iter)

        # Update the positions using the SMA equation
        for j in range(SearchAgents_no):
            if random.random() < w:
                # Approach food (exploitation)
                Best_pos = positions[0]
                positions[j] = positions[j] + np.random.rand() * (Best_pos - positions[j])
            else:
                # Explore randomly
                random_index = random.randint(0, SearchAgents_no - 1)
                random_pos = positions[random_index]
                positions[j] = positions[j] + z * np.random.rand() * (random_pos - positions[j])

            # Vibrate randomly
            positions[j] = positions[j] + vb * np.random.randn(dim)

            # Ensure positions stay within bounds
            positions[j] = np.clip(positions[j], lb, ub)

      # Get the best solution
    Best_pos = positions[0]
    Best_score = objf(Best_pos)

    return Best_pos, Best_score

def fitness_function_factory_SMA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
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
        model_path = 'SMA_best_model_min_val_loss.pth'
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']
        return loss

    def fitness_func_SMA(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        return solution_fitness
    return fitness_func_SMA

#------------------------------------------------- SMA optimization----------------------------------------------------------
def intermediate_fusion_SMA(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, SearchAgents_no, criterion):
    fitness_func_SMA = fitness_function_factory_SMA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    dim = sum(1 for data_type in loaders_dict.keys() for i in loaders_dict[data_type]) + 1         # Number of solutions
    lb, ub = 1, 10 # Lower and upper bounds
    solution, fitness = SMA(fitness_func_SMA, lb, ub, dim, SearchAgents_no, max_iter)
    return np.around(solution).astype(int), fitness