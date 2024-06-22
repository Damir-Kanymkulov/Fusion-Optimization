import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import binary_repr
from early_fusion import MLP

#-------------------------------------------Definition of functions---------------------------------------------------------
class NewMyEnsemble(nn.Module):
    def __init__(self, model_list, n_layers = 5, output_dim=1):
        super(NewMyEnsemble, self).__init__()
        self.model_list = model_list
        input_dim = 0
        for i in range(len(self.model_list)):
            input_dim += list(self.model_list[i].children())[0][-2].out_features

        if n_layers == None:
            layer_list = nn.ModuleList()
            in_nodes = input_dim
            if (input_dim // 2) % 2 != 0:
                out_nodes = input_dim // 2 + 1
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
                in_nodes = out_nodes
            else:
                out_nodes = input_dim // 2
            while out_nodes % 2 == 0 and (in_nodes // 2) != 1 and out_nodes != 0:
                out_nodes = in_nodes//2
                layer_list.append(nn.Linear(in_nodes, out_nodes))
                layer_list.append(nn.ReLU())
                in_nodes = out_nodes
            self.layers = nn.Sequential(*layer_list)
            out_nodes = out_nodes
        else:
            layer_list = nn.ModuleList()
            nodes = np.linspace(input_dim, 1, num=n_layers + 1, dtype=int)
            for idx in range(len(nodes) - 2):
                nodes[idx], nodes[idx + 1]
                layer_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
                layer_list.append(nn.ReLU())
            self.layers = nn.Sequential(*layer_list)
            out_nodes = nodes[-2]
        self.output_fc = nn.Linear(out_nodes, output_dim)

    def forward(self, x_list):
        models = []
        for i in range(len(self.model_list)):
            model = self.model_list[i](x_list[i].float())
            models.append(model)
        x = torch.cat(models, dim=1)  # .detach() Detaching models, so models will not be updated
        x = self.layers(x)
        output = self.output_fc(x)
        return output

def new_train_one_epoch_intermediate(model, optimizer, train_loaders, criterion, device):
    model.train()
    loss_step = []
    num_loaders = len(train_loaders)
    for data in zip(*[train_loaders[i] for i in range(num_loaders)]):
        inputs, labels =  [data[i][0][0].to(device) for i in range(num_loaders)], [data[i][1][0].to(device) for i in range(num_loaders)]
        # # inputs1, labels1, inputs2, labels2 = inputs1.double(), labels1, inputs2.double(), labels2
        outputs = model(inputs)
        loss = criterion(outputs, labels[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_step.append(loss.item())
    loss_curr_epoch = np.mean(loss_step)
    return loss_curr_epoch

@torch.no_grad()
def new_validate_intermediate(model, val_loaders, criterion, device):
    model.eval()
    loss_step = []
    num_loaders = len(val_loaders)
    for data in zip(*[val_loaders[i] for i in range(num_loaders)]):
        inputs, labels =  [data[i][0][0].to(device) for i in range(num_loaders)], [data[i][1][0].to(device) for i in range(num_loaders)]
        # inputs1, labels1, inputs2, labels2 = inputs1.double(), labels1, inputs2.double(), labels2
        outputs = model(inputs)
        val_loss = criterion(outputs, labels[0])
        loss_step.append(val_loss.item())
    val_loss_epoch = torch.tensor(loss_step).mean().numpy()
    return val_loss_epoch

def new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, path):
    dict_log = {"train_loss": [], "val_loss": []}
    best_val_loss = 1e8
    model = model.to(device)
    for epoch in range(num_epochs):
        loss_curr_epoch = new_train_one_epoch_intermediate(model, optimizer, train_loaders, criterion, device)
        val_loss = new_validate_intermediate(model, val_loaders, criterion, device)
        # Print epoch results to screen
        print(f'Ep {epoch + 1}/{num_epochs}: Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
        dict_log["train_loss"].append(loss_curr_epoch)
        dict_log["val_loss"].append(val_loss)
        # Use this code to save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f'{path}', epoch, optimizer, val_loss)
    return dict_log

def save_model(model, path, epoch, optimizer, val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        }, path)

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model

#-------------------------------------------Main part---------------------------------------------------------
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
    print("Combination of layers: {}".format(solution))
    dict_log = new_train_intermediate(model, optimizer, num_epochs, train_loaders, val_loaders, criterion, device, model_path)
    checkpoint = torch.load(model_path)
    loss = checkpoint['loss']
    return loss

def intermediate_fusion_brute_force_search(dimension_dict, loaders_dict, device, lr=0.01, num_epochs=1, criterion=nn.L1Loss()):
    ub = 2
    num_solutions = sum(1 for data_type in loaders_dict.keys() for i in loaders_dict[data_type]) + 1
    combinations = 2 ** num_solutions
    orig_sol = np.ones(num_solutions, dtype=int)
    best_loss = 1e8
    best_solution = orig_sol
    best_combo = {'loss': best_loss, 'solution': best_solution}
    for i in range(ub - 1):
        for combination in range(combinations):
            solution_arr = binary_repr(combination, width = len(binary_repr(combinations)) - 1)
            solution_arr = list(solution_arr)
            solution = orig_sol + [int(sol) for sol in solution_arr]
            loss = calculate_loss(dimension_dict, loaders_dict, solution, device, lr, num_epochs, criterion)
            if loss < best_loss:
                best_loss = loss
                best_combo['loss'] = best_loss
                best_combo['solution'] = solution
        orig_sol = orig_sol + np.ones(num_solutions, dtype=int)
    return best_combo['loss'], best_combo['solution']