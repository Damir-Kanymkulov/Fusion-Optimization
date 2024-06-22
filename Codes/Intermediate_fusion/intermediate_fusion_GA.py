import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pygad
from early_fusion import MLP
from intermediate_fusion_brute_force_search import NewMyEnsemble, new_train_intermediate

#-------------------------------------------Definition of functions---------------------------------------------------------
def fitness_function_factory_GA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
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
                model = MLP(input_dim=input_dim, n_layers=solution[index], fusion="intermediate")
                models.append(model)
                index += 1
            train_loaders.extend(train_loaders_per_type)
            val_loaders.extend(val_loaders_per_type)
        model = NewMyEnsemble(models, n_layers=solution[index])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #Training
        model_path = 'best_intermediate_fusion_model.pth'
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']
        return loss

    def fitness_func(ga_instance, solution, solution_idx):
        loss = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        solution_fitness = 1 / (loss + 1e-6)
        return solution_fitness
    return fitness_func

def callback_generation_GA(ga_instance):
    print("Generation: {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Best solution = {fitness}".format(fitness=ga_instance.best_solution()[0]))

#------------------------------------------------- GA optimization----------------------------------------------------------
def intermediate_fusion_GA(dimension_dict, loaders_dict, device, lr, num_epochs, num_generations, criterion):
    num_genes = sum(1 for data_type in loaders_dict.keys() for i in loaders_dict[data_type]) + 1
    ga_instance = pygad.GA(num_generations=num_generations,
                    num_parents_mating=3,
                    fitness_func=fitness_function_factory_GA(dimension_dict, loaders_dict, device, lr, num_epochs, criterion),
                    on_generation=callback_generation_GA,
                    sol_per_pop=3,
                    num_genes=num_genes, # Num of solutions (one for each model)
                    gene_type=int,
                    gene_space=[(1, 10) for i in range(num_genes)],  #  Range for number of layers for each model
                    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    loss = 1/(solution_fitness)
    return solution, loss