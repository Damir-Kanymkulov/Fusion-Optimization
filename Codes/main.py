import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from Shopee_dataset import dimension_dict, loaders_dict, train_shopee_combined_loader, val_shopee_combined_loader #train_combined_loader, val_combined_loader
from early_fusion import early_fusion
from intermediate_fusion_brute_force_search import intermediate_fusion_brute_force_search
from intermediate_fusion_GA import intermediate_fusion_GA
from intermediate_fusion_GWO import intermediate_fusion_GWO
from intermediate_fusion_PSO import intermediate_fusion_PSO
from intermediate_fusion_SMA import intermediate_fusion_SMA
from late_fusion import late_fusion

#---------------------------------------------------------------Fusion results------------------------------------------------------------------------
if __name__ == "__main__":
    #Hyperparameters
    num_epochs = 10
    lr = 0.01
    criterion=nn.L1Loss()
    num_generations = 3 # The number of iterations in GA
    Max_iter = 3 # The number of iterations in GWO
    SearchAgents_no = 3 # Number of wolves seeking value in GWO
    num_particles = 3 # The number of particles in PSO
    max_iter = 3 # The number of iterations in PSO
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("------------------------------------------------------------Training--------------------------------------------------------------------------")
    #Training fusion models
    print("-----------------------------------------------------Start of early fusion--------------------------------------------------------------------")
    early_fusion_loss = early_fusion(dimension_dict, train_shopee_combined_loader, val_shopee_combined_loader, device, lr, num_epochs, criterion)
    print("-------------------------------------Start of intermediate fusion with brute-force search-----------------------------------------------------")
    intermediate_brute_force_loss, solution = intermediate_fusion_brute_force_search(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    print("---------------------------------------------Start of intermediate fusion with GA-------------------------------------------------------------")
    solution_GA, intermediate_fusion_loss_GA = intermediate_fusion_GA(dimension_dict, loaders_dict, device, lr, num_epochs, num_generations, criterion)
    print("---------------------------------------------Start of intermediate fusion with GWO------------------------------------------------------------")
    solution_GWO, intermediate_fusion_loss_GWO = intermediate_fusion_GWO(dimension_dict, loaders_dict, device, lr, num_epochs, Max_iter, SearchAgents_no, criterion)
    print("---------------------------------------------Start of intermediate fusion with PSO------------------------------------------------------------")
    solution_PSO, intermediate_fusion_loss_PSO = intermediate_fusion_PSO(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, num_particles, criterion)
    print("---------------------------------------------Start of intermediate fusion with SMA------------------------------------------------------------")
    solution_SMA, intermediate_fusion_loss_SMA = intermediate_fusion_SMA(dimension_dict, loaders_dict, device, lr, num_epochs, max_iter, SearchAgents_no, criterion)
    print("-----------------------------------------------------Start of late fusion---------------------------------------------------------------------")
    late_fusion_loss = late_fusion(dimension_dict, loaders_dict, device, lr, num_epochs, criterion)
    print("------------------------------------------------------------Results---------------------------------------------------------------------------")
    print("Early fusion loss: {:.5f}".format(early_fusion_loss))
    print("Brute force search loss: {:.5f} \t Best combination of NN layers: {}".format(intermediate_brute_force_loss, solution))
    print("Intermediate fusion loss (GA optimization): {:.5f} \t Best combination of NN layers: {}".format(intermediate_fusion_loss_GA, solution_GA))
    print("Intermediate fusion loss (GWO optimization): {:.5f} \t Best combination of NN layers: {}".format(intermediate_fusion_loss_GWO, solution_GWO))
    print("Intermediate fusion loss (PSO optimization): {:.5f} \t Best combination of NN layers: {}".format(intermediate_fusion_loss_PSO, solution_PSO))
    print("Intermediate fusion loss (SMA optimization): {:.5f} \t Best combination of NN layers: {}".format(intermediate_fusion_loss_SMA, solution_SMA))
    print("Late fusion loss: {:.5f}".format(late_fusion_loss))
    print("----------------------------------------------------------------------------------------------------------------------------------------------")