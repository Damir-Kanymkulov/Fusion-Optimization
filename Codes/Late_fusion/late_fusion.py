import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from early_fusion import MLP, train

#----------------------------------------------------Late fusion---------------------------------------------------------------
def late_fusion(dimension_dict, loaders_dict, device, lr=0.01, num_epochs=1, criterion=nn.L1Loss()):
    loss_array = []
    for data_type in loaders_dict.keys():
        for i, (train_loader, val_loader) in enumerate(loaders_dict[data_type]):
            model = MLP(input_dim=dimension_dict[data_type], fusion="late")
            optimizer = optim.Adam(model.parameters(), lr=lr)
            #Training
            model_path = 'best_late_fusion_model_' + data_type + '_' + str(i) + '.pth'
            num_dict_log = train(model, optimizer, num_epochs, train_loader, val_loader, criterion, device, model_path)
            checkpoint = torch.load(model_path)
            loss_array.append(checkpoint['loss'])
    #Results of late fusion
    loss = sum(loss_array) / len(loss_array)
    return loss