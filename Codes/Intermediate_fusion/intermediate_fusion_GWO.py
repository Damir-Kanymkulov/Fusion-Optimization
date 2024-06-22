import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.optim as optim
import random
import numpy
from early_fusion import MLP
from intermediate_fusion_brute_force_search import NewMyEnsemble, new_train_intermediate

#-------------------------------------------Definition of functions---------------------------------------------------------
def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    #=== initialization  alpha, beta, and delta_pos=======
    Alpha_pos = numpy.zeros(dim)  #  Location . formation 30 A list of
    Alpha_score = float("inf")  #  This means “ Plus or minus infinity ”, All the numbers are equal to  +inf  Small ; It's just infinite ：float("inf");  Negative infinity ：float("-inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")  # float()  Function to convert integers and strings to floating-point numbers .

    #====list List the type =============
    if not isinstance(lb, list):  #  effect ： To determine whether an object is a known type . Its first parameter （object） As object , The second parameter （type） For type name , If the type of the object is the same as that of parameter 2, return True
        lb = [lb] * dim  #  Generate [100,100,.....100]30 individual
    if not isinstance(ub, list):
        ub = [ub] * dim

    #======== Initialize the location of all wolves ===================
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  #  formation 5*30 Number [-100,100) within
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[
            i]  #  formation [5 individual 0-1 Number of numbers ]*100-（-100）-100
    Convergence_curve = numpy.zeros(Max_iter)

    #======== Iterative optimization =====================
    for l in range(0, Max_iter):  #  iteration 1000
        for i in range(0, SearchAgents_no):  # 5
            #==== Returns the search agent that exceeds the boundary of the search space ====
            for j in range(dim):  # 30
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])  # clip This function limits the elements in the array to a_min(-100), a_max(100) Between , Greater than a_max That makes it equal to  a_max, Less than a_min, That makes it equal to a_min.

            #=== Calculate the objective function of each search agent ==========
            fitness = objf(Positions[i, :])  #  Bring a row of data into function calculation
            # print(" After calculation ：",fitness)

            #==== to update  Alpha, Beta, and Delta================
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        #=========== In the above cycle ,Alpha、Beta、Delta===========
        a = 2 - l * ((2) / Max_iter);  # a from 2 Linear reduction to 0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1] Mainly generate one 0-1 Is a random floating point number .
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  # Equation (3.3)
                C1 = 2 * r2;  # Equation (3.4)
                # D_alpha Indicates the candidate wolf and Alpha The distance of the wolf
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[
                    i, j]);  # abs()  Function returns the absolute value of a number .Alpha_pos[j] Express Alpha Location ,Positions[i,j]) Location of candidate gray wolf
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1 Express basis alpha The position vector of the next generation gray wolf

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                Positions[i, j] = (X1 + X2 + X3) / 3  #  The location of the candidate wolf is updated according to Alpha、Beta、Delta The address of the next generation of gray wolf .

        Convergence_curve[l] = Alpha_score;

        if (l % 1 == 0):
            print([' The number of iterations is ' + str(l) + '  Iterative results of ' + str(Alpha_score)]);  #  Results of each iteration
    print("The best optimal value of the objective funciton found by GWO is: {:.5f} \t Best solution: {}".format(Alpha_score, numpy.around(Alpha_pos).astype(int)))
    return Alpha_pos, Alpha_score

def fitness_function_factory_GWO(dimension_dict, loaders_dict, device, lr, num_epochs, criterion):
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
        model_path = 'GWO_best_model_min_val_loss.pth'
        dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
        checkpoint = torch.load(model_path)
        loss = checkpoint['loss']
        return loss

    def fitness_func_GWO(solution):
        solution_fitness = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
        return solution_fitness
    return fitness_func_GWO

#------------------------------------------------- GWO optimization----------------------------------------------------------
def intermediate_fusion_GWO(dimension_dict, loaders_dict, device, lr, num_epochs, Max_iter, SearchAgents_no, criterion):
    fitness_func_GWO = fitness_function_factory_GWO(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    dim = sum(1 for data_type in loaders_dict.keys() for i in loaders_dict[data_type]) + 1         # Search range of wolf
    lb, ub = 1, 10       # Lower and upper bounds
    solution, solution_fitness = GWO(fitness_func_GWO, lb, ub, dim, SearchAgents_no, Max_iter)
    return numpy.around(solution).astype(int), solution_fitness