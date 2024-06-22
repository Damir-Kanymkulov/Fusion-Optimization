import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#-----------------------------------------------------Definition of functions-----------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, fusion, n_layers=None, output_dim=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fusion = fusion
        self.output_dim = output_dim
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
            self.out_nodes = out_nodes
        else:
            layer_list = nn.ModuleList()
            nodes = np.linspace(input_dim, 1, num=n_layers + 2, dtype=int)
            for idx in range(len(nodes) - 2):
                nodes[idx], nodes[idx + 1]
                layer_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
                layer_list.append(nn.ReLU())
            self.layers = nn.Sequential(*layer_list)
            self.out_nodes = nodes[-2]

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.layers(x.float())
        if self.fusion == "intermediate":
            return x
        if self.fusion == "early" or self.fusion ==  "late":
            output_fc = nn.Linear(self.out_nodes, self.output_dim)
            y_pred = output_fc(x)
            return y_pred
        else:
            raise ValueError("Incorrect fusion mode!!!")

def train_one_epoch(model, optimizer, train_loader, criterion, device):
  model.train()
  loss_step = []
  for inputs, labels in train_loader:
    optimizer.zero_grad()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels.unsqueeze(1))
    loss.backward()
    optimizer.step()
    loss_step.append(loss.item())
  loss_curr_epoch = np.mean(loss_step)
  return loss_curr_epoch

def validate(model, val_loader, criterion, device):
  model.eval()
  loss_step = []
  for inputs, labels in val_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    val_loss = criterion(outputs, labels.unsqueeze(1))
    loss_step.append(val_loss.item())
  val_loss_epoch = torch.tensor(loss_step).mean().numpy()
  return val_loss_epoch

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

def train(model, optimizer, num_epochs, train_loader, val_loader, criterion, device, path):
  dict_log = {"train_loss": [], "val_loss": []}
  best_val_loss = 1e8
  model = model.to(device)
  for epoch in range(num_epochs):
    loss_curr_epoch = train_one_epoch(model, optimizer, train_loader, criterion, device=device)
    val_loss = validate(model, val_loader, criterion, device)
    # Print epoch results to screen
    print(f'Ep {epoch + 1}/{num_epochs}: Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
    dict_log["train_loss"].append(loss_curr_epoch)
    dict_log["val_loss"].append(val_loss)
    # Use this code to save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, f'{path}', epoch, optimizer, val_loss)
  return dict_log

#----------------------------------------------------Early fusion---------------------------------------------------------------
def early_fusion(dimension_dict, train_loader, val_loader, device, lr=0.01, num_epochs=1, criterion=nn.L1Loss()):
    input_dim = sum(list(dimension_dict.values()))
    model = MLP(input_dim=input_dim, fusion="early")
    optimizer = optim.Adam(model.parameters(), lr=lr)
#Training
    model_path = 'best_early_fusion_model.pth'
    dict_log = train(model, optimizer, num_epochs, train_loader, val_loader, criterion, device, model_path)
#Results of early fusion
    checkpoint = torch.load('best_early_fusion_model.pth')
    loss = checkpoint['loss']
    return loss