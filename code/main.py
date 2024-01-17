import copy
import json
import os

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import myutils as mu
from myutils import QuickDataset
from myutils import TrainableNet
from myutils import MNIST_FF_Net
from myutils import MNIST_CNN_Net
from myutils import CIFAR_CNN_Net
from myutils import ActivationNetwork
from myutils import ActivationSingleLayer


 
# TODO LIST:
# VERSION 2 (MULTI-MODEL TRAINING)
# DONE. Implement uniform_zip() in myutils.py for datasets with different number of batches
# DONE. Rework TrainableNet optimizer handling
# 3.    Rework seeds handling
# 4.    Make a script to update v1 experiments to v2 & update eval_exp.py and eval_multiexp.py to work with v2
#
# VERSION 3 (MORE ARCHITECTURE FREEDOM)
# 4. Allow for multiple trainable activation functions in a sinlge network and more complex activation networks with different base activation functions
#    and reflect it in the metadata



### DEVICE ###
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


### SEED DEFINITION ###
# TODO: Allow for non-None seed
    # NOTE: Actual code is a weird mix, from the first implementation of single-model training, which allowed for non-None seed
    #       and also didn't take into account experiment replication even if seed was None
    # IDEA: Generate a seed if SEED is None, and use that seed to generate all the other seeds (the ones saved in metadata)
SEED = None


### DATASET DEFINITION ###
# DATASET = 'MNIST'
# DATASET = 'FashionMNIST'
DATASET = 'CIFAR10'
# DATASETS = ['MNIST', 'FashionMNIST']
BATCH_SIZE = 128


### BEST ACTIVATIONS ###
BEST_MODELS_FOLDERS = ['2_hidden_units', '4_hidden_units', '8_hidden_units', '16_hidden_units', '32_hidden_units', '64_hidden_units', 'PReLU']
# BEST_MODELS_MNIST = ['MNIST_MiniNetwork_2_231104.0558', 'MNIST_MiniNetwork_4_231104.0613', 'MNIST_MiniNetwork_8_231104.1050', 'MNIST_MiniNetwork_16_231104.140', 'MNIST_MiniNetwork_32_231104.1735', 'MNIST_MiniNetwork_64_231104.2105', 'MNIST_PReLU_231103.2356']
# BEST_MODELS_FASHION_MNIST = ['FashionMNIST_Mininetwork_2_231107.0059', 'FashionMNIST_Mininetwork_4_231107.0437', 'FashionMNIST_Mininetwork_8_231107.0820', 'FashionMNIST_Mininetwork_16_231107.1035', 'FashionMNIST_Mininetwork_32_231107.1410', 'FashionMNIST_Mininetwork_64_231107.1655', 'FashionMNIST_MNIST_PReLU_231107.2235']
BEST_MODELS_MNIST_CNN = ['MNIST_CNN_Mininetwork_2_231207.1422', 'MNIST_CNN_Mininetwork_4_231207.1538', 'MNIST_CNN_Mininetwork_8_231207.2107', 'MNIST_CNN_Mininetwork_16_231208.0127', 'MNIST_CNN_Mininetwork_32_231208.1302', 'MNIST_CNN_Mininetwork_64_231208.2217', 'MNIST_CNN_PReLU_231212.0025']
BEST_MODELS_FASHION_MNIST_CNN = ['FashionMNIST_CNN_Mininetwork_2_231209.0701', 'FashionMNIST_CNN_Mininetwork_4_231209.1207', 'FashionMNIST_CNN_Mininetwork_8_231209.1409', 'FashionMNIST_CNN_Mininetwork_16_231209.1809', 'FashionMNIST_CNN_Mininetwork_32_231210.0147', 'FashionMNIST_CNN_Mininetwork_64_231210.1756', 'FashionMNIST_CNN_PReLU_231213.0718']
BEST_MODELS_10_MNIST = ['MNIST_Mininetwork_2_231121.1945', 'MNIST_Mininetwork_4_231121.2149', 'MNIST_Mininetwork_8_231122.0309', 'MNIST_Mininetwork_16_231122.0839', 'MNIST_Mininetwork_32_231122.1035', 'MNIST_Mininetwork_64_231122.1433', 'MNIST_CNN_PReLU_231217.2145']
BEST_MODELS_10_FASHION_MNIST = ['FashionMNIST_Mininetwork_2_231122.1840', 'FashionMNIST_Mininetwork_4_231123.0020', 'FashionMNIST_Mininetwork_8_231123.0608', 'FashionMNIST_Mininetwork_16_231123.0954', 'FashionMNIST_Mininetwork_32_231123.1357', 'FashionMNIST_Mininetwork_64_231123.1601', 'FashionMNIST_CNN_PReLU_231218.0304']
# BEST_MODELS_EXPS = BEST_MODELS_MNIST_CNN + BEST_MODELS_FASHION_MNIST_CNN + BEST_MODELS_10_MNIST + BEST_MODELS_10_FASHION_MNIST
# BEST_MODELS_PARENTS = ['mnist_cnn_exps']*7 + ['fashion_cnn_exps']*7 + ['mnist_10_multi_model_exps']*7 + ['fashion_10_multi_model_exps']*7
BEST_MODELS = [os.path.join(model, exp) for model, exp in zip(BEST_MODELS_FOLDERS, BEST_MODELS_FASHION_MNIST_CNN)]


### ACTIVATIONS TO EVALUATE ###
NUM_UNITS = [2,4,8,16,32,64]
ACTIVATIONS = [ActivationNetwork(num_units=i) for i in NUM_UNITS] + [nn.ReLU(), nn.PReLU()]
# ACTIVATIONS = [nn.ReLU(), nn.PReLU()]
# ACTIVATIONS = [ActivationNetwork(num_units=i) for i in NUM_UNITS]


### MAIN EXPERIMENT FOLDER ###
# MAIN_EXP_DIR = './experiments/fashion_convergence_exps'
# MAIN_EXP_DIR = './experiments/fashion_cnn_exps'
# MAIN_EXP_DIRS = ['./experiments/mnist_10_multi_model_exps', './experiments/fashion_10_multi_model_exps']
MAIN_EXP_DIRS = ['./experiments/cifar_exploration_exps']


### EXPERIIMENTS FOLDER FOR EACH ACTIVATION ###
EXPERIMENT_FOLDERS = [f'{i}_hidden_units' for i in NUM_UNITS] + ['ReLU', 'PReLU']
# EXPERIMENT_FOLDERS = ['ReLU', 'PReLU']
# EXPERIMENT_FOLDERS = [f'{i}_hidden_units' for i in NUM_UNITS]
# EXPERIMENT_FOLDERS = [f'best_{i}_hidden_units' for i in NUM_UNITS] + ['best_PReLU']
# EXPERIMENT_FOLDERS = ['convergence_exps/best_PReLU', 'fashion_convergence_exps/best_PReLU']
# EXPERIMENT_FOLDERS = [f'best_{i}_hidden_units' for i in NUM_UNITS] + ['best_PReLU']


### INDIVIDUAL EXPERIMENT FOLDER NAME ###
# EXPERIMENT_NAMES = [f'{DATASET}__Mininetwork_{i}' for i in NUM_UNITS] + [f'{DATASET}_ReLU', f'{DATASET}_PReLU']
# EXPERIMENT_NAMES = [f'{DATASET}_CNN_Mininetwork_{i}' for i in NUM_UNITS] + [f'{DATASET}_CNN_ReLU', f'{DATASET}_CNN_PReLU']
# EXPERIMENT_NAMES = [f'{DATASET}_CNN_ReLU', f'{DATASET}_CNN_PReLU']
# EXPERIMENT_NAMES = [f'{DATASET}_Mininetwork_{i}' for i in NUM_UNITS]
# EXPERIMENT_NAMES = [f'MIX_5_5_Mininetwork_{i}' for i in NUM_UNITS]
# EXPERIMENT_NAMES = [f'{DATASET}_Mininetwork_{i}' for i in NUM_UNITS] + [f'{DATASET}_PReLU']
EXPERIMENT_NAMES = [f'{DATASET}_CNN_Mininetwork_{i}' for i in NUM_UNITS] + [f'{DATASET}_CNN_ReLU', f'{DATASET}_CNN_PReLU']
# EXPERIMENT_NAMES = [f'{dataset}_PReLU' for dataset in DATASET]
# EXPERIMENT_NAMES = [f'MIX_5_5_ReLU', f'MIX_5_5_PReLU']


### NUMBER OF ITERATIONS ###
NUM_MODELS = 1
EPOCHS = 30
NUM_EXPERIMENTS = 1


### EXPERIMENT CONFIGURATION ###
# Cross validation with same initialization for each fold:
# EXPERIMENT_CONFIG = {
#     'val_fold': 5,
#     'cross_val': True,
#     'fold_same_init': True,
#     'epochs': EPOCHS}

# Cross validation with random initialization for each fold:
# EXPERIMENT_CONFIG = {
#     'val_fold': 5,
#     'cross_val': True,
#     'fold_same_init': False,
#     'epochs': EPOCHS}

# No cross validation:
# EXPERIMENT_CONFIG = {
#     'val_fold': 5,
#     'cross_val': False,
#     'fold_same_init': False,
#     'epochs': EPOCHS}

# No validation at all:
EXPERIMENT_CONFIG = {
    'val_fold': 0,
    'cross_val': False,
    'fold_same_init': False,
    'epochs': EPOCHS}



# TODO: Maybe rework for cleanliness, but dont sacrifice flexibility
def main():

    # Set seed
    if SEED is not None:
        torch.manual_seed(SEED)

    for MAIN_EXP_DIR in MAIN_EXP_DIRS:

        # Load dataset
        # mnist_dataset = QuickDataset('MNIST', batch_size=BATCH_SIZE, seed=SEED)
        # fashion_dataset = QuickDataset('FashionMNIST', batch_size=BATCH_SIZE, seed=SEED)
        cifar_dataset = QuickDataset('CIFAR10', batch_size=BATCH_SIZE, seed=SEED)
        datasets = cifar_dataset

        print(cifar_dataset.train_dataset[0][0].shape)

        for activation, folder, exp_name in zip(ACTIVATIONS, EXPERIMENT_FOLDERS, EXPERIMENT_NAMES):
        # for activation, folder, exp_name in zip(ACTIVATIONS, EXPERIMENT_FOLDERS, EXPERIMENT_NAMES):

            # Load dataset
            # dataset = QuickDataset(dataset_name, batch_size=BATCH_SIZE, seed=SEED)
            
            # Define save directory
            save_exp_dir = os.path.join(MAIN_EXP_DIR, folder)

            # Load best experiment
            # metadata, models, histories = mu.load_experiment(best_exp, exp_dir=MAIN_EXP_DIR, verbose=False)
            # history = histories[0]
            # model = models[0]
            # if 'version' in metadata.keys():
            #     history = history[0]

            # model.load_state_dict(history['checkpoint'][np.argmin(history['val_loss'])])

            # Extract activation and freeze parameters
            # activation = model.net.activation

            # for param in activation.parameters():
            #     param.requires_grad = False

            # Launch experiments
            for _ in range(NUM_EXPERIMENTS):

                # Reset activation parameters:
                if hasattr(activation, 'reset_parameters'):
                    activation.reset_parameters()           
                
                # Define optimizer:
                # optimizer_name, optimizer_args = 'SGD', {'lr': 0.001, 'momentum': 0.9}
                optimizer_name, optimizer_args = 'Adam', {}

                # Define criterium:
                criterium = nn.CrossEntropyLoss(reduction='sum')

                # Create networks to train
                nets = []

                core_net = mu.CIFAR_CNN_Net_test(activation)
                # core_net = MNIST_FF_Net(activation)
                nets.append(TrainableNet(core_net, criterium, optimizer=optimizer_name, optimizer_args=optimizer_args))

                act_optimizer = nets[0].act_optimizer
                act_optimizer_args = nets[0].act_optimizer_args

                for _ in range(NUM_MODELS - 1):
                    core_net = MNIST_FF_Net(activation)
                    nets.append(TrainableNet(core_net, criterium, optimizer=optimizer_name, optimizer_args=optimizer_args, act_optimizer=act_optimizer, act_optimizer_args=act_optimizer_args))

                # Train network:
                metadata, models, histories, valid = train(datasets, nets, **EXPERIMENT_CONFIG, device=DEVICE)

                # Save experiment to disk
                if valid:
                    mu.save_experiment(exp_name, metadata, models, histories, exp_dir=save_exp_dir)


def train(datasets, networks, epochs=1, val_fold=5, cross_val=False, fold_same_init=False, seed=None, device=torch.device('cpu')):
    valid_exp = True # Flag to interrupt experiment if marked as invalid

    if not isinstance(networks, list):
        networks = [networks]

    if not isinstance(datasets, list):
        datasets = [datasets]

    if len(networks) != len(datasets):
        if len(datasets) != 1:
            raise ValueError('Number of networks and datasets must be equal or number of datasets must be 1')
        else:
            datasets = datasets * len(networks)
        
    assert val_fold != 1, 'val_fold must be either 0 or > 1'
    

    # Move networks to device
    for network in networks:
        network.to(device)


    metadata = {
        'version': 'v2',

        # TODO: Update whith seeds rework
        'seed': seed,

        'task': {
            'dataset_name': [dataset.name for dataset in datasets],
            'batch_size': [dataset.batch_size for dataset in datasets],

            'criterium': [network.loss_fun.__class__.__name__ for network in networks],

            'net_optimizer': [network.optimizer_name for network in networks],
            'net_optimizer_args': [network.optimizer_args for network in networks],
            'act_optimizer': [network.act_optimizer_name for network in networks],
            'act_optimizer_args': [network.act_optimizer_args for network in networks],
        },

        'architecture': {
            'net_name': [network.net.__class__.__name__ for network in networks],

            # NOTE: Assumes a single trainable activation function
            'activation_name': networks[0].net.activation.__class__.__name__,
            'activation_args': networks[0].net.activation.metadata if hasattr(networks[0].net.activation, 'metadata') else {},
        },

        'experiment': {
            'epochs': epochs,

            'val_fold': val_fold,
            'cross_val': cross_val,
            'fold_same_init': fold_same_init,
        },

        'results': {}
        }
    
    print(json.dumps(metadata, indent=4))
    print('')


    # Save networks & optimizers initial states
    # TODO: FIX. Commented because act_optimizer can be None
    # net_initial_states = [copy.deepcopy(network.state_dict()) for network in networks]
    # opt_initial_states = [(copy.deepcopy(network.net_optimizer.state_dict()),
    #                        copy.deepcopy(network.act_optimizer.state_dict()))
    #                              for network in networks]


    # Load dataloaders
        # First dimension is for different folds of cross validation
        # Second dimension is for the different models in multi-model training
        # Third dimension is for train / validation
    if val_fold != 0:
        tv_dataloaders = [dataset.getDataLoader(train=True, val_fold=val_fold, cross_val=cross_val) for dataset in datasets]
    else:
        tv_dataloaders = [[(dataset.getDataLoader(train=True), dataset.getDataLoader(train=False))] for dataset in datasets]

    tv_dataloaders = [list(e) for e in zip(*tv_dataloaders)] # Transpose first and second dimensions


    ### MAIN TRAINING LOOP ###
    fold_histories = []
    for fold, dataloaders in enumerate(tv_dataloaders):
        train_dataloaders = [e[0] for e in dataloaders]
        val_dataloaders = [e[1] for e in dataloaders]

        # Reset networks & optimizers for a fresh start for each fold
        # NOTE: This piece of code is outdated, but needed if using cross validation.
        # TODO: Fix toghether with seeds rework
        # if seed is None and not fold_same_init:
        #     for layer in network.children():
        #         if hasattr(layer, 'reset_parameters'):
        #             layer.reset_parameters()
        #     # TODO: Reset optimizer parameters too
        # else:       
        #     network.load_state_dict(net_initial_state)
        #     network.optimizer.load_state_dict(opt_initial_state)


        # Print fold info (if relevant)
        if cross_val:
            print(f'\nFold {fold + 1}/{val_fold}\n')

        
        # TRAINING AND VALIDATION
        histories = [{'epoch':[], 'train_loss':[], 'val_loss':[] , 'train_acc':[], 'val_acc':[], 'checkpoint':[]} for _ in range(len(networks))]
        for i, network in enumerate(networks):
            histories[i]['initial_state'] = copy.deepcopy(network.state_dict())

        for epoch in range(epochs):

            # Interrupt experiment if marked as invalid
            if not valid_exp:
                return metadata, networks, fold_histories, valid_exp


            # Training
            train_losses = [0.0]*len(networks)
            train_correct = [0]*len(networks)
            total = np.max([len(train_dataloader) for train_dataloader in train_dataloaders])
            for batches in tqdm(mu.uniform_zip(*train_dataloaders), leave=False, desc=f'Epoch {epoch+1} training', total=total):

                # Find last non-None batch index
                for i, batch in enumerate(batches[::-1]):
                    if batch is not None:
                        last_batch = len(batches) - i - 1
                        break

                for i, network in enumerate(networks):
                    if batches[i] is None:
                        continue

                    inputs, labels = batches[i]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Step on activation only once, for the last model with a non-None batch
                    step_on = 'both' if i == last_batch else 'net'

                    loss = network.train_step(inputs, labels, step_on=step_on)
                    train_losses[i] += loss

                    correct = network.test_step(inputs, labels, test_fun=mu.correct_classifications)
                    train_correct[i] += correct


            # Validation (or directly evaluation if no validation)
            val_losses = [0.0]*len(networks)
            val_correct = [0]*len(networks)
            total = np.max([len(val_dataloader) for val_dataloader in val_dataloaders])
            for batches in tqdm(mu.uniform_zip(*val_dataloaders), leave=False, desc=f'Epoch {epoch+1} validation', total=total):
                for i, network in enumerate(networks):
                    if batches[i] is None:
                        continue

                    inputs, labels = batches[i]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    loss = network.test_step(inputs, labels)
                    val_losses[i] += loss

                    correct = network.test_step(inputs, labels, test_fun=mu.correct_classifications)
                    val_correct[i] += correct

            
            # Save history
            for i in range(len(networks)):
                histories[i]['epoch'].append(epoch + 1)
                histories[i]['train_loss'].append(train_losses[i] / len(train_dataloaders[i].sampler))
                histories[i]['train_acc'].append(train_correct[i] / len(train_dataloaders[i].sampler))
                histories[i]['val_loss'].append(val_losses[i] / len(val_dataloaders[i].sampler))
                histories[i]['val_acc'].append(val_correct[i] / len(val_dataloaders[i].sampler))

                histories[i]['checkpoint'].append(copy.deepcopy(networks[i].state_dict()))


            # Print epoch info
            for i in range(len(networks)):
                train_loss = histories[i]['train_loss'][-1]
                train_loss_diff = train_loss - histories[i]['train_loss'][-2] if epoch > 0 else 0
                val_loss = histories[i]['val_loss'][-1]
                val_loss_diff = val_loss - histories[i]['val_loss'][-2] if epoch > 0 else 0
                train_acc = histories[i]['train_acc'][-1]
                train_acc_diff = train_acc - histories[i]['train_acc'][-2] if epoch > 0 else 0
                val_acc = histories[i]['val_acc'][-1]
                val_acc_diff = val_acc - histories[i]['val_acc'][-2] if epoch > 0 else 0

                print(f'[E{epoch+1}, M{i+1}]\ttrain_loss: {train_loss:.3f} ({train_loss_diff:+.3f}), val_loss: {val_loss:.3f} ({val_loss_diff:+.3f}), train_acc: {train_acc:.3f} ({train_acc_diff:+.3f}), val_acc: {val_acc:.3f} ({val_acc_diff:+.3f})')


            # Invalidation condition
            threshold_acc = 0.2
            threshold_epoch = 2
            if epoch >= threshold_epoch:
                train_accs_thresh = [history['train_acc'][-1] < threshold_acc for history in histories]
                if any(train_accs_thresh):
                    valid_exp = False
                    break


        # MODEL SELECTION & EVALUATION
        if val_fold != 0:
            for i, network in enumerate(networks):

                # Load test dataloader
                test_dataloader = datasets[i].getDataLoader(train=False)


                # Model selection (last epoch)
                last_epoch = len(histories[i]['epoch']) - 1
                network.load_state_dict(histories[i]['checkpoint'][last_epoch])

                # Evaluation for last model
                last_test_loss = 0.0
                last_test_correct = 0
                for batch in tqdm(test_dataloader, leave=False, desc=f'Model {i+1} test (last epoch)', total=len(test_dataloader)):
                    inputs, labels = batch

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    loss = network.test_step(inputs, labels)
                    last_test_loss += loss

                    correct = network.test_step(inputs, labels, test_fun=mu.correct_classifications)
                    last_test_correct += correct


                # Model selection (minimum validation loss)
                min_val_epoch = np.argmin(histories[i]['val_loss'])
                network.load_state_dict(histories[i]['checkpoint'][min_val_epoch])

                # Evaluation for best model
                best_test_loss = 0.0
                best_test_correct = 0
                for batch in tqdm(test_dataloader, leave=False, desc=f'Model {i+1} test (best epoch)', total=len(test_dataloader)):
                    inputs, labels = batch

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    loss = network.test_step(inputs, labels)
                    best_test_loss += loss

                    correct = network.test_step(inputs, labels, test_fun=mu.correct_classifications)
                    best_test_correct += correct


                # Save history
                histories[i]['last_epoch'] = last_epoch + 1
                histories[i]['last_test_loss'] = last_test_loss / len(test_dataloader.dataset)
                histories[i]['last_test_acc'] = last_test_correct / len(test_dataloader.dataset)

                histories[i]['best_epoch'] = min_val_epoch + 1
                histories[i]['best_test_loss'] = best_test_loss / len(test_dataloader.dataset)
                histories[i]['best_test_acc'] = best_test_correct / len(test_dataloader.dataset)


                # Print evaluation info (only for best model)
                val_loss = histories[i]['val_loss'][min_val_epoch]
                test_loss = histories[i]['best_test_loss']
                val_acc = histories[i]['val_acc'][min_val_epoch]
                test_acc = histories[i]['best_test_acc']

                if i == 0: print('')
                print(f'[M{i+1}]\tbest_epoch: {min_val_epoch+1}, val_loss: {val_loss:.3f} test_loss: {test_loss:.3f}, val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f}')
                if i == len(networks) - 1 and cross_val: print('')


        # SAVE RESULTS
        fold_histories.append(histories)
    

    # Add results to metadata
    mean_best_epochs = []
    
    mean_train_losses_best = []
    mean_train_accs_best = []
    mean_val_losses_best = []
    mean_val_accs_best = []
    mean_test_losses_best = []
    mean_test_accs_best = []

    mean_train_losses_last = []
    mean_train_accs_last = []
    mean_val_losses_last = []
    mean_val_accs_last = []
    mean_test_losses_last = []
    mean_test_accs_last = []

    for i in range(len(networks)):

        mean_best_epoch = np.mean([np.argmin(histories[i]['val_loss']) for histories in fold_histories])

        mean_train_loss_best = np.mean([histories[i]['train_loss'][np.argmin(histories[i]['val_loss'])] for histories in fold_histories])
        mean_train_acc_best = np.mean([histories[i]['train_acc'][np.argmin(histories[i]['val_loss'])] for histories in fold_histories])
        mean_val_loss_best = np.mean([histories[i]['val_loss'][np.argmin(histories[i]['val_loss'])] for histories in fold_histories])
        mean_val_acc_best = np.mean([histories[i]['val_acc'][np.argmin(histories[i]['val_loss'])] for histories in fold_histories])

        mean_train_loss_last = np.mean([histories[i]['train_loss'][-1] for histories in fold_histories])
        mean_train_acc_last = np.mean([histories[i]['train_acc'][-1] for histories in fold_histories])
        mean_val_loss_last = np.mean([histories[i]['val_loss'][-1] for histories in fold_histories])
        mean_val_acc_last = np.mean([histories[i]['val_acc'][-1] for histories in fold_histories])


        mean_best_epochs.append(mean_best_epoch)

        mean_train_losses_best.append(mean_train_loss_best)
        mean_train_accs_best.append(mean_train_acc_best)
        mean_val_losses_best.append(mean_val_loss_best)
        mean_val_accs_best.append(mean_val_acc_best)

        mean_train_losses_last.append(mean_train_loss_last)
        mean_train_accs_last.append(mean_train_acc_last)
        mean_val_losses_last.append(mean_val_loss_last)
        mean_val_accs_last.append(mean_val_acc_last)

            
        if val_fold != 0: # If there proper validation was performed (and therefore model selection and evaluation was performed)
            mean_test_loss_best = np.mean([histories[i]['best_test_loss'] for histories in fold_histories])
            mean_test_acc_best = np.mean([histories[i]['best_test_acc'] for histories in fold_histories])

            mean_test_loss_last = np.mean([histories[i]['last_test_loss'] for histories in fold_histories])
            mean_test_acc_last = np.mean([histories[i]['last_test_acc'] for histories in fold_histories])


            mean_test_losses_best.append(mean_test_loss_best)
            mean_test_accs_best.append(mean_test_acc_best)

            mean_test_losses_last.append(mean_test_loss_last)
            mean_test_accs_last.append(mean_test_acc_last)


        # Print all folds mean results if relevant    
        if cross_val:
            if val_fold == 0:
                print(f'[M{i+1}]\tmean_train_acc_best: {mean_train_acc_best:.3f}, mean_val_acc_best: {mean_val_acc_best:.3f}')
                print(f'[M{i+1}]\tmean_train_acc_last: {mean_train_acc_last:.3f}, mean_val_acc_last: {mean_val_acc_last:.3f}')
            else:
                print(f'[M{i+1}]\tmean_train_acc_best: {mean_train_acc_best:.3f}, mean_val_acc_best: {mean_val_acc_best:.3f}, mean_test_acc_best: {mean_test_acc_best:.3f}')
                print(f'[M{i+1}]\tmean_train_acc_last: {mean_train_acc_last:.3f}, mean_val_acc_last: {mean_val_acc_last:.3f}, mean_test_acc_last: {mean_test_acc_last:.3f}')

            if i == len(networks) - 1: print('')


    metadata['results']['mean_best_epoch'] = mean_best_epochs

    metadata['results']['mean_train_loss_best'] = mean_train_losses_best
    metadata['results']['mean_train_acc_best'] = mean_train_accs_best
    metadata['results']['mean_val_loss_best'] = mean_val_losses_best
    metadata['results']['mean_val_acc_best'] = mean_val_accs_best
    metadata['results']['mean_test_loss_best'] = mean_test_losses_best
    metadata['results']['mean_test_acc_best'] = mean_test_accs_best

    metadata['results']['mean_train_loss_last'] = mean_train_losses_last
    metadata['results']['mean_train_acc_last'] = mean_train_accs_last
    metadata['results']['mean_val_loss_last'] = mean_val_losses_last
    metadata['results']['mean_val_acc_last'] = mean_val_accs_last
    metadata['results']['mean_test_loss_last'] = mean_test_losses_last
    metadata['results']['mean_test_acc_last'] = mean_test_accs_last


    # Move networks back to cpu
    for network in networks:
        network.to(torch.device('cpu'))

    return metadata, networks, fold_histories, valid_exp


if __name__ == "__main__":
    main()