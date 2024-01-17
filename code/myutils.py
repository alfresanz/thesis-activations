import os
import json
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms





# ------------------------------------------------------------------------------------------





# TODO: Add possibility of data augmentation
class QuickDataset:
    def __init__(self, name, **kwargs):
        self.name = name
        self.load()
        
        # Load configuration
        self.batch_size = kwargs.get('batch_size', 128)
        self.seed = kwargs.get('seed', None)

    # TODO: Maybe rework the transforms
    def load(self):
        if self.name == 'MNIST':
            transform = transforms.ToTensor()
            # NOTE: No need for normalization, it already loads normalized images

            self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            self.test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

        elif self.name == 'FashionMNIST':
            transform = transforms.ToTensor()
            # NOTE: No need for normalization, it already loads normalized images

            self.train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            self.test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

        elif self.name == 'CIFAR10':
            # TO-COMPLETE?
            transform = transforms.ToTensor()

            self.train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            self.test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        else:
            print("Unknown dataset!")
            exit()

    def getDataLoader(self, train=True, val_fold=0, cross_val=False):
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset

        if val_fold == 0 or not train:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return dataloader
        
        else:
            if cross_val == False:
                train_dataset, val_dataset = data.random_split(dataset, [1-1/val_fold, 1/val_fold])

                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
                return [(train_dataloader, val_dataloader)]
            
            else:
                kfold = KFold(n_splits=val_fold, shuffle=True, random_state=self.seed)

                dataloaders = []
                for train_ids, val_ids in kfold.split(dataset):
                    train_subsampler = SubsetRandomSampler(train_ids)
                    val_subsampler = SubsetRandomSampler(val_ids)

                    train_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_subsampler)
                    val_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_subsampler)

                    dataloaders.append((train_dataloader, val_dataloader))

                return dataloaders





# ------------------------------------------------------------------------------------------





class TrainableNet(nn.Module):
    def __init__(self, net, loss_fun, optimizer='Adam', act_optimizer=None, optimizer_args={}, act_optimizer_args=None):
        super(TrainableNet, self).__init__()

        self.net = net
        self.loss_fun = loss_fun

        # NOTE: The activation optimizer will copy as much info as possible from the main optimizer when no overriding congiguration is given
        if act_optimizer is None:
            act_optimizer = optimizer

            if act_optimizer_args is None:
                act_optimizer_args = optimizer_args

        elif isinstance(act_optimizer, optim.Optimizer):
            if act_optimizer_args is None:
                print(f'For a pre-instantiated optimizer for the activation, please supply the act_optimizer_args.')
                exit()

        else:
            if act_optimizer_args is None:
                act_optimizer_args = {}

        self.net_optimizer = None
        self.optimizer_name = optimizer
        self.optimizer_args = optimizer_args

        if isinstance(act_optimizer, optim.Optimizer):
            self.act_optimizer = act_optimizer
            self.act_optimizer_name = act_optimizer.__class__.__name__
        else:
            self.act_optimizer = None
            self.act_optimizer_name = act_optimizer

        self.act_optimizer_args = act_optimizer_args

        self.set_optimizers()
    
    def load_optimizer(optimizer_name, optimizer_args, parameters):
        if len(parameters) == 0:
            return None

        if optimizer_name == 'Adam':
            return optim.Adam(parameters, **optimizer_args)
        elif optimizer_name == 'SGD':
            return optim.SGD(parameters, **optimizer_args)
        else:
            print(f'Unknown optimizer {optimizer_name}!')
            exit()

    def set_optimizers(self, new_act_optimizer=False):
        all_parameters = list(self.net.parameters())
        activation_parameters = list(self.net.activation.parameters())

        network_parameters = []
        for param in all_parameters:
            is_activation_param = False
            for act_param in activation_parameters:
                if param is act_param:
                    is_activation_param = True
                    break
            
            if not is_activation_param:
                network_parameters.append(param)

        self.net_optimizer = TrainableNet.load_optimizer(self.optimizer_name, self.optimizer_args, network_parameters)

        if self.act_optimizer is None or new_act_optimizer:
            self.act_optimizer = TrainableNet.load_optimizer(self.act_optimizer_name, self.act_optimizer_args, activation_parameters)

    def reset(self, new_act_optimizer=None, new_act_optimizer_args=None):
        for layer in self.net.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        if new_act_optimizer is None:
            self.set_optimizers(new_act_optimizer=True)
        else:
            if new_act_optimizer_args is None:
                print(f'For a pre-instantiated optimizer for the activation, please supply the act_optimizer_args.')
                exit()

            self.act_optimizer = new_act_optimizer
            self.act_optimizer_name = new_act_optimizer.__class__.__name__
            self.act_optimizer_args = new_act_optimizer_args

            self.set_optimizers(new_act_optimizer=False)

    def forward(self, x):
        return self.net(x)
    
    def train_step(self, inputs, labels, step_on='both'):
        assert step_on in ['net', 'act', 'both', 'none'], "step_on must be either 'net', 'act', 'both' or 'none'"
        
        self.train()

        outputs = self(inputs)
        loss = self.loss_fun(outputs, labels)
        loss.backward()

        if step_on == 'net' or step_on == 'both':
            self.net_optimizer.step()
            self.net_optimizer.zero_grad()
        if step_on == 'act' or step_on == 'both':
            if self.act_optimizer is not None:
                self.act_optimizer.step()
                self.act_optimizer.zero_grad()

        return loss.item()

    def test_step(self, inputs, labels, test_fun=None):
        self.eval()

        outputs = self(inputs)

        if test_fun is None:
            loss = self.loss_fun(outputs, labels)
            return loss.item()
        
        return test_fun(outputs, labels)





# ------------------------------------------------------------------------------------------





class ElementwiseLinearLayer(nn.Module):

    # NOTE: This class acts as the individual layers of an ActivationNetwork
    # which is meant to act on the input feature vector elementwise.
    # 'input_units' and 'output units' represent the hidden size within the 
    # ActivationNetwork and must be 1 for the first and last layers respectively.
    def __init__(self, input_units, output_units):
        super(ElementwiseLinearLayer, self).__init__()

        # NOTE: The first two dimensions of the weights and biases correspond to 
        # the batch size and the dimension of the hidden features vector. Since we
        # want to act elementwise, we keep thos as 1 (using tensor broadcasting)
        # and use the two next dimensions for operations.
        self.w = torch.nn.Parameter(torch.empty(1, 1, input_units, output_units))
        self.b = torch.nn.Parameter(torch.empty(1, 1, output_units))
        self.reset_parameters()

    # NOTE: Weight initialization from nn.Linear
    # (https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear)
    def reset_parameters(self):

        # Weights 
        nn.init.kaiming_uniform_(self.w, a=np.sqrt(5))

        # Biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    # NOTE: Expects an input of shape (batch_size, num_features, input_units, 1).
    # The output will be of shape (batch_size, num_features, output_units, 1).
    def forward(self, x):
        x = torch.sum(self.w*x, dim=-2) + self.b
        return x[..., None]





class BaseActivationNetwork(nn.Module):
    def __init__(self, num_units=32, base_activation=F.relu):
        super(BaseActivationNetwork, self).__init__()

        self.activation = base_activation
        self.num_units = num_units
        self.metadata = {'base_activation': base_activation.__name__, 'num_units': num_units}

    def get_function(self, range=(-5, 5), num_points=1000):
        x = torch.tensor(np.linspace(range[0], range[1], num=num_points))
        y = self(x).detach().numpy()[0]
        x = x.detach().numpy()

        return (x,y)





class ActivationNetwork(BaseActivationNetwork):
    def __init__(self, num_units=32, base_activation=F.relu):
        # Work with the general case of multiple hidden layers, that is, num_units is a list
        if not isinstance(num_units, list):
            num_units = [num_units]
        
        super(ActivationNetwork, self).__init__(num_units=num_units, base_activation=base_activation)

        # Define the layers of the network
        first_layer = ElementwiseLinearLayer(1, num_units[0])
        hidden_layers = [ElementwiseLinearLayer(num_units[i], num_units[i+1]) for i in range(len(num_units)-1)]
        last_layer = ElementwiseLinearLayer(num_units[-1], 1)

        self.layers = nn.ModuleList([first_layer] + hidden_layers + [last_layer])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        # Add two extra dimensions for proper elementwise application of ActivationNetwork
        x = x[..., None, None] 

        # Apply the each layer and followed by the base activation function
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        x = self.layers[-1](x)

        # Remove the two extra dimensions (which should be dummy dimensions by now)
        return x[..., 0, 0]





# Original ActivationLayer class from Luis' code
class ActivationSingleLayer(BaseActivationNetwork):
    def __init__(self, num_units=32, base_activation=F.relu):
        super(ActivationSingleLayer, self).__init__(num_units=num_units, base_activation=base_activation)

        self.w1 = torch.nn.Parameter(torch.randn(1, 1, num_units))
        self.w2 = torch.nn.Parameter(torch.randn(1, 1, num_units))
        self.b1 = torch.nn.Parameter(torch.zeros(1, 1, num_units))
        self.b2 = torch.nn.Parameter(torch.tensor(0.0))

    def reset_parameters(self):
        nn.init.uniform_(self.w1)
        nn.init.uniform_(self.w2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x):
        x = x[:, :, None]
        x = self.activation(self.w1*x + self.b1)
        # x = torch.sum(self.w2*x + self.b2, dim=2) # NOTE: Network output explodes with SGD{'lr': 0.001, 'momentum': 0.9}
        x = torch.sum(self.w2*x, dim=2) + self.b2 # NOTE: More stable, still explodes sometimes
        return x
    




# ------------------------------------------------------------------------------------------





# Feed forward network
class MNIST_FF_Net(nn.Module):
    def __init__(self, activation):
        super(MNIST_FF_Net, self).__init__()

        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 10)

        self.activation = activation

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x





# Convolutional network
# TODO: Compare number of parameters with the FF network
# TODO: Make sure dimensions are correct
class MNIST_CNN_Net(nn.Module):
    def __init__(self, activation):
        super(MNIST_CNN_Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.fc = nn.Linear(7*7*32, 10)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool(x)
        x = self.activation(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(-1, 7*7*32)
        x = self.fc(x)
        return x





# Convolutional network
# TODO: Compare number of parameters with the FF network
# TODO: Make sure dimensions are correct
class CIFAR_CNN_Net(nn.Module):
    def __init__(self, activation):
        super(CIFAR_CNN_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.fc = nn.Linear(8*8*32, 10)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool(x)
        x = self.activation(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(-1, 8*8*32)
        x = self.fc(x)
        return x
    




# Convolutional network
# TODO: Compare number of parameters with the FF network
# TODO: Make sure dimensions are correct
class CIFAR_CNN_Net_test(nn.Module):
    def __init__(self, activation):
        super(CIFAR_CNN_Net_test, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.fc = nn.Linear(4*4*64, 10)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool(x)
        x = self.activation(self.conv2(x))
        x = self.maxpool(x)
        x = self.activation(self.conv3(x))
        x = self.maxpool(x)

        x = x.view(-1, 4*4*64)
        x = self.fc(x)
        return x





# ------------------------------------------------------------------------------------------





# outputs: batch_size x num_classes (values from -inf to +inf)
# labels: batch_size (values between 0 and num_classes-1)
def correct_classifications(outputs, labels, pred_selection='max'):
    
    # Select predicted class
    if pred_selection == 'max':
        _, predicted = torch.max(outputs, 1)
    else:
        print("Unknown prediction selection method!")
        exit()
    
    # Count correct predictions
    correct = (predicted == labels).sum().item()
    
    return correct





# outputs: batch_size x num_classes (values from -inf to +inf)
# labels: batch_size (values between 0 and num_classes-1)
def confussion_matrix(outputs, labels, pred_selection='max'):
    # Get number of classes
    num_classes = outputs.shape[1]

    # Initialize confusion matrix
    cf_matrix = np.zeros((num_classes, num_classes))
    
    # Select predicted class
    if pred_selection == 'max':
        _, predicted = torch.max(outputs, 1)
    else:
        print("Unknown prediction selection method!")
        exit()

    # Populate confusion matrix
    for i in range(len(predicted)):
        cf_matrix[labels[i], predicted[i]] += 1
    
    return cf_matrix





# ------------------------------------------------------------------------------------------





def save_experiment(exp_name, metadata, models, histories, exp_dir='./experiments'):
    # Check if experiment directory exists
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    
    # Get current date and time
    now = datetime.now()
    str_date_time = now.strftime('%y%m%d.%H%M')

    # Create experiment directory
    os.mkdir(f'{exp_dir}/{exp_name}_{str_date_time}')    

    # Save metadata to text file
    with open(f'{exp_dir}/{exp_name}_{str_date_time}/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    # Save model definition to file
    for i, model in enumerate(models):
        torch.save(model, f'{exp_dir}/{exp_name}_{str_date_time}/model_{i}.pt')

    # Save history to pickle file
    with open(f'{exp_dir}/{exp_name}_{str_date_time}/history.pkl', 'wb') as f:
        pickle.dump(histories, f)





def load_experiment(exp_name, exp_dir='./experiments', old=0, verbose=True):
    # Check if experiment directory exists
    if not os.path.exists(exp_dir):
        print("Experiment directory does not exist!")
        exit()

    # Check if experiment exists
    if not os.path.exists(f'{exp_dir}/{exp_name}'):
        experiments = os.listdir(exp_dir)
        experiments = [exp for exp in experiments if exp.startswith(exp_name)]

        if len(experiments) == 0:
            print("Experiment does not exist!")
            exit()
        else:
            exp_name = experiments[-1-old]

    if verbose:
        print(f'Loading {exp_name}\n\n')

    # Load metadata
    with open(f'{exp_dir}/{exp_name}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    all_files = os.listdir(f'{exp_dir}/{exp_name}')
    
    # Load models
    models = [torch.load(f'{exp_dir}/{exp_name}/{file}') for file in all_files if file.startswith('model')]

    # Load histories
    history_file = [file for file in all_files if file.startswith('history')][0] # TODO: Remove when updating format of experiments
    with open(f'{exp_dir}/{exp_name}/{history_file}', 'rb') as f:
        histories = pickle.load(f)
    
    return metadata, models, histories





def get_datetime(exp_name):
    date_str = exp_name.split('_')[-1]
    return datetime.strptime(date_str, '%y%m%d.%H%M')

def get_metadata(exp_name, exp_dir='./experiments'):
    with open(f'{exp_dir}/{exp_name}/metadata.json', 'r') as f:
        metadata = json.load(f)
    return metadata





# ------------------------------------------------------------------------------------------





def uniform_zip(*lists):
    lists = [iter(l) for l in lists]

    lengths = np.array([len(l) for l in lists])
    max_length = np.max(lengths)

    count = np.zeros(len(lists))
    for _ in range(max_length):
        count += lengths
        to_yield = count >= max_length

        yield tuple([next(l) if y else None for l, y in zip(lists, to_yield)])

        count -= to_yield * max_length