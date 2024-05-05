#!/usr/bin/env python
# coding: utf-8

#### Single-Fidelity AvgNorm GNN

##################
import os, random, sys, math, joblib

import megnet

# Get the path of the megnet module
module_path = megnet.__file__

# Print the module path
print("Path to the megnet module:", module_path)

import os, json, time
import logging
import numpy as np
import pandas as pd

from scipy.stats import moment
from scipy.stats import wasserstein_distance 
from scipy.linalg import eig
from scipy.linalg import inv 
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr

from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.core.periodic_table import Element

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
from tensorflow.keras.layers import Add, Concatenate, Input, Dense, Conv1D, Dropout, Embedding, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint


from megnet.layers import MEGNetLayer, Set2Set
from megnet.models import MEGNetModel, GraphModel
from megnet.data.graph import EmbeddingMap, GaussianDistance
from megnet.data.crystal import CrystalGraph
from monty.serialization import dumpfn


from operator import itemgetter

from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read, write

from scipy.signal import find_peaks

# Update Optuna to the latest version
import optuna
##################


loss_weights = [1.0, 3.7]

##################
# Best hyps
n_feat_atom = 22
nfeat_bond = 180
n_feat_global = 2
n1 = 400; n2 = 300; n3 = 150
n4 = 150; n5 = 75
n6 = 75; n7 = 400
activ = 'elu'
dropout_rate = 0.05
batch_size = 110


n_epochs = 320
lr_ = 1e-3
decay_factor_ = 0.02
# Define convergence threshold and patience
convergence_threshold = 1e-06  # Adjust this threshold as needed
convergence_patience = 40  # Number of epochs with no significant improvement to consider convergence
##################


print(sys.path)

print("TensorFlow version:", tf.__version__)

# Check if eager execution is enabled
if tf.executing_eagerly():
    print("Eager execution is enabled.")
else:
    print("Eager execution is not enabled (using non-eager execution).")
##################

with open('./dataset/mbj_avg_data.json', 'r') as f:
    data = json.load(f)
dff = pd.DataFrame(data)
##################     


# convert cif to pymatgen structures
def cif2pystructure(row):
    py_struc = Structure.from_str(row['cif'], fmt='cif')
    return py_struc

# Apply the function to create the 'pystructure' column
dff['pystructure'] = dff.apply(cif2pystructure, axis=1)
######################


def filter_by_number_density(df):
    # Calculate number density for each row
    df['number_density'] = [df['pystructure'].iloc[i].num_sites / df['pystructure'].iloc[i].volume
                            for i in range(df.shape[0])]
    # Drop rows with 'number_density' < 0.005
    df_filtered = df[df['number_density'] >= 0.005]
    # Adjust the index numbering
    df_filtered.index = list(np.arange(0, df_filtered.shape[0]))
    return df_filtered

# Apply the function to dff
dff = filter_by_number_density(dff)
######################

# Make a column for abs_coeff
dff['abs_interp'] = dff.apply(
    lambda row: -np.array(row['real_avg_interp']) + \
    np.sqrt(np.array(row['real_avg_interp'])**2 + \
            np.array(row['imag_avg_interp'])**2), axis=1)


# Function to find the index of the closest value in energy_range to the band_gap
def find_closest_index(row):
    return np.abs(np.array(row['energy_range']) - np.array(row['band_gap'])).argmin()

# Apply the function to each row and create the 'band_gap_index' column
dff['band_gap_index'] = dff.apply(find_closest_index, axis=1)


def set_values_to_zero(row):
    band_gap_index = row['band_gap_index']
    abs_interp = row['abs_interp']

    # Set values with index <= band_gap_index to zero
    row['abs_interp'] = np.where(
        np.arange(len(
            abs_interp)) <= band_gap_index, 0, abs_interp)

    return row

# Apply the function to each row
dff = dff.apply(set_values_to_zero, axis=1)
##################


# Specify a list of columns you want to include in the sub-DataFrame
selected_columns = ['pystructure', 'band_gap', 'band_gap_index', 'energy_range', 
                    'abs_interp']

# Create the sub-DataFrame from the main DataFrame
df = dff[selected_columns]
##################

# Count non-None values in each column
non_none_count = df.count()

print('count:', non_none_count)
##################


#----------------#
##################
# #### Normalize to norm & normalized spectrum
# nomalize as discrete probability distribution
def normalize_target(vector): # spectrum normalized as probability distribution
    n_vec = vector / np.sum(vector) 
    return n_vec.tolist()

def target_norm(vector): # by target norm, we refer average value over the spectrum
    norm = np.sum(vector)/len(vector)
    return norm 


df['abs_normalized'] = df['abs_interp'].apply(normalize_target)
df['abs_norm'] = df['abs_interp'].apply(target_norm)

##################
#----------------#


##################
# #### Split into train/test

# split df into df_train, df_val, and df_test
def train_validate_test_split(df, train_percent=0.80, validate_percent=0.05, seed=42):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    df_train = df.iloc[perm[:train_end]]
    df_val = df.iloc[perm[train_end:validate_end]]
    df_test = df.iloc[perm[validate_end:]]
    return df_train, df_val, df_test

# train/test split
df_train, df_test, df_left = train_validate_test_split(df)


# Adjust the index numbering 
df_train.index = list(np.arange(0,df_train.shape[0]))
#df_val.index = list(np.arange(0,df_val.shape[0]))
df_test.index = list(np.arange(0,df_test.shape[0]))


structures_train = df_train['pystructure'].tolist()
norms_train = np.expand_dims(np.array(df_train['abs_norm'].tolist()), axis = -1)
normalized_vector_train =  np.array(df_train['abs_normalized'].tolist())
targets_train = np.concatenate((norms_train, normalized_vector_train), axis=1)
###
###
structures_test = df_test['pystructure'].tolist()
norms_test =  np.expand_dims(np.array(df_test['abs_norm'].tolist()), axis = -1)
normalized_vector_test =  np.array(df_test['abs_normalized'].tolist())     
targets_test = np.concatenate((norms_test, normalized_vector_test), axis=1)

print('targets vector shape: ', targets_test.shape[1])
##################

# Define energy grid size
energy_grid_delta = 0.04
# Size of the multioutput spectrum points
pointwise_spectrum_shape = df['abs_interp'].iloc[0].shape[0]
print('pointwise_spectrum_shape = ', pointwise_spectrum_shape)
##################
# #### Get the energy grid
en_grid = df['energy_range'].tolist()
delta_e = en_grid[0][1] - en_grid[0][0]
energy_grid_size = pointwise_spectrum_shape
##################



##################
# Define keras model generator
def build_keras_model(n_feat_atom, nfeat_bond, n_feat_global, activ, dropout_rate, energy_grid_size, 
                      n1, n2, n3, n4, n5, n6, n7):

    # use z number as the only atom feature
    #x1 = Input(shape=(None, n_feat_atom), dtype='int32', 
    #           name="atom_int_input") # atom feature placeholder (only z as feature using one-hot encoding)
    #x1_ = x1

    # use a learnable embedding as atom feature
    x1 = Input(shape=(None,), dtype='int32', name="atom_int_input")
    x1_ = Embedding(input_dim=95, output_dim=n_feat_atom, name="atom_embedding")(x1)


    x2 = Input(shape=(None, nfeat_bond), dtype='float32', 
               name="bond_float_input") # bond feature placeholder (bond attributes are float distances)
    x2_ = x2
    
    
    x3 = Input(shape=(None, n_feat_global), dtype='float32',
               name="state_float_input") # global feature placeholder
    x3_ = x3

    
    x4 = Input(shape=(None,), dtype='int32', name="bond_index_1_input") # bond index1 placeholder
    x5 = Input(shape=(None,), dtype='int32', name="bond_index_2_input") # bond index2 placeholder
    x6 = Input(shape=(None,), dtype='int32', name="atom_graph_index_input") # atom_ind placeholder
    x7 = Input(shape=(None,), dtype='int32', name="bond_graph_index_input") # bond_ind placeholder


    # gather inputs
    inputs = [x1, x2, x3, x4, x5, x6, x7]

    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##

    # pass the inputs to feedforward dense layers
    a1 = Dense(n1, activation = activ, name="atom_ff_layer_1-1")(x1_)
    x1_ = Dense(n2, activation= activ, name="atom_ff_layer_1-2")(a1)

    b1 = Dense(n1, activation= activ, name="bond_ff_layer_1-1")(x2_)
    x2_ = Dense(n2, activation= activ, name="bond_ff_layer_1-2")(b1)
    
    # Identity layer
    x3_ = Dense(2, activation= activ, name="global_ff_layer_1", trainable=False)(x3_)

    # Pass to output of the feedforward dense layers to the MEGNetLayer layer
    # the megnet_output is a tuple of new graphs E, V and u
    megnet_out_1 = MEGNetLayer([n1, n2, n3], [n1, n2, n3], [n4, n5], pool_method='mean', 
                             activation= activ)(([x1_, x2_, x3_, x4, x5, x6, x7]))
    x1_ = megnet_out_1[0]
    x2_ = megnet_out_1[1]
    x3_ = megnet_out_1[2]

    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##

    # pass the inputs to feedforward dense layers
    a2 = Dense(n1, activation= activ, name="atom_ff_layer_2-1")(x1_)
    x1_ = Dense(n2, activation= activ, name="atom_ff_layer_2-2")(a2)

    b2 = Dense(n1, activation= activ, name="bond_ff_layer_2-1")(x2_)
    x2_ = Dense(n2, activation= activ, name="bond_ff_layer_2-2")(b2)

    x3_ = Dense(n4, activation= activ, name="global_ff_layer_2")(x3_)

    # Pass to output of the feedforward dense layers to the MEGNetLayer layer
    # the megnet_output is a tuple of new graphs E, V and u
    megnet_out_2 = MEGNetLayer([n1, n2, n3], [n1, n2, n3], [n4, n5], pool_method='mean', 
                             activation= activ)(([x1_, x2_, x3_, x4, x5, x6, x7]))
    x1_ = megnet_out_2[0]
    x2_ = megnet_out_2[1]
    x3_ = megnet_out_2[2]

    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##

    # pass the inputs to feedforward dense layers
    a3 = Dense(n1, activation= activ, name="atom_ff_layer_3-1")(x1_)
    x1_ = Dense(n2, activation= activ, name="atom_ff_layer_3-2")(a3)

    b3 = Dense(n1, activation= activ, name="bond_ff_layer_3-1")(x2_)
    x2_ = Dense(n2, activation= activ, name="bond_ff_layer_3-2")(b3)

    x3_ = Dense(n4, activation= activ, name="global_ff_layer_3")(x3_)

    # Pass to output of the feedforward dense layers to the MEGNetLayer layer
    # the megnet_output is a tuple of new graphs E, V and u
    megnet_out_3 = MEGNetLayer([n1, n2, n3], [n1, n2, n3], [n4, n5], pool_method='mean', 
                             activation= activ)(([x1_, x2_, x3_, x4, x5, x6, x7]))
    x1_ = megnet_out_3[0]
    x2_ = megnet_out_3[1]
    x3_ = megnet_out_3[2]

    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##
    
    x1_ = Dropout(dropout_rate, name=f"dropout_atom_3")(x1_, training = True)
    x2_ = Dropout(dropout_rate, name=f"dropout_bond_3")(x2_, training= True)
    x3_ = Dropout(dropout_rate, name=f"dropout_state_3")(x3_, training= True)

    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##

    # set2set for both the atom and bond
    npass = 3

    node_vec = Set2Set(T=npass, n_hidden=n6, name="set2set_atom")([x1_, x6])
    # print('Node vec', node_vec)
    edge_vec = Set2Set(T=npass, n_hidden=n6, name="set2set_bond")([x2_, x7])

    # concatenate atom, bond, and global
    # After the readout, the atomic, bond, and state vectors are concatenated and passed through
    # multilayer perceptrons to generate the final output
    # note: x4_ and x5_ are not used
    final_vec = Concatenate(axis=-1)([node_vec, edge_vec, x3_])
    #final_vec = Dropout(dropout_rate, name="dropout_final")(final_vec, training= True)

    # final dense layers
    final_vec = Dense(n7, activation= activ, name="readout_0")(final_vec)
    final_vec = Dense(n7, activation= activ, name="readout_1")(final_vec)

    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##

    # define outputs
    
    y1_pred = Dense(1, activation='relu', name="output_norms")(final_vec)
    
    y2_pred = Dense(energy_grid_size, activation= 'linear', name="output_normalized")(final_vec)
    # Add the softmax layer 
    y2_pred = Dense(energy_grid_size, activation='softmax', name="logits_normalized")(y2_pred)
    
    # Concatenate the two outputs
    outputs = Concatenate(axis=-1)([y1_pred, y2_pred])

    # define the inputs and outputs to the model
    keras_model = Model(inputs=inputs, outputs=outputs)

    return keras_model
##################


##################
# #### graph converter
# Define graph_converter_generator
r_cutoff = 5.5; gaussian_width = 0.5

def build_graph_converter(r_cutoff, gaussian_width, nfeat_bond):
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)

    # graph_converter is an object that turns a structure to a graph
    # GaussianDistance is a distance converter that expands the distance with Gaussian basis set
    graph_converter = CrystalGraph(cutoff=r_cutoff, 
                                   bond_converter = GaussianDistance(gaussian_centers, gaussian_width))
    
    return graph_converter
##################


##################
# #### Load data to the model
# Function to prepapre data generators
def prepare_data_generators(graph_model, 
                            structures_train, targets_train, structures_test, targets_test, 
                            batch_size):

    # For train, val, and test sets,  compute the graphs from structures and spit out (graphs, targets)
    structure_graphs_train, target_graphs_train = graph_model.get_all_graphs_targets(
        structures_train, targets_train)
    #structure_graphs_val, target_graphs_val = graph_model.get_all_graphs_targets(structures_val, targets_val)
    structure_graphs_test, target_graphs_test = graph_model.get_all_graphs_targets(
        structures_test, targets_test)

    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##                                            


    # train_inputs is a tuple of 6 lists .. the first 5 lists are lists of lists of structure 
    # attributes ('atom', 'bond', 'state', 'index1', 'index2') .. and the 6th list is a list of the target arrays
    train_inputs = graph_model.graph_converter.get_flat_data(
        structure_graphs_train, target_graphs_train)  
    test_inputs = graph_model.graph_converter.get_flat_data(
        structure_graphs_test, target_graphs_test) 
    
    ## ------------------------------------------------------ ##
    ## ------------------------------------------------------ ##
    
    # get the train generator for keras fitting
    train_generator = graph_model._create_generator(*train_inputs, batch_size=batch_size)
    test_generator = graph_model._create_generator(*test_inputs, batch_size=batch_size)
    
    return train_generator, test_generator
##################


##################
# Define your custom loss function with additional arguments
def custom_loss_wrapper(energy_grid_size, energy_grid_delta, loss_weights):
    def custom_loss(y_true, y_pred):

        y1_pred = y_pred[:, :, :1]
        y2_pred = y_pred[:, :, 1:]

        y1_true = y_true[:, :, :1]
        y2_true = y_true[:, :, 1:]
    
        ## ----------------- ##
        component1 = kb.mean(kb.abs((y1_pred - y1_true)), axis=-1) # average value loss (norm loss)
        ## ----------------- ##    
        component2 = 1000 * kb.mean(kb.abs(y2_pred - y2_true), axis=-1) # Spectrum point-wise loss 

        ## ----------------- ##
        ## ----------------- ##
        ## ----------------- ##

        # Define loss components and their weights 
        loss_components = [component1, component2]
        
        
        total_loss = sum(loss_components[i] * loss_weights[i] for i in range(len(loss_components)))

    
        return total_loss

    return custom_loss
##################


##################
# Define your custom loss function with additional arguments
def metric_1_wrapper(energy_grid_size, energy_grid_delta, loss_weights):
    def component_1(y_true, y_pred):

        y1_pred = y_pred[:, :, :1]
        y2_pred = y_pred[:, :, 1:]

        y1_true = y_true[:, :, :1]
        y2_true = y_true[:, :, 1:]
    
        ## ----------------- ##
        component1 = kb.mean(kb.abs((y1_pred - y1_true)), axis=-1) # norm loss
    
        return component1

    return component_1


# Define your custom loss function with additional arguments
def metric_2_wrapper(energy_grid_size, energy_grid_delta, loss_weights):
    def component_2(y_true, y_pred):

        y1_pred = y_pred[:, :, :1]
        y2_pred = y_pred[:, :, 1:]

        y1_true = y_true[:, :, :1]
        y2_true = y_true[:, :, 1:]
    
        ## ----------------- ##
        component2 = 1000 * kb.mean(kb.abs(y2_pred - y2_true), axis=-1) # normalized spectrum loss 
    
        return component2

    return component_2


##################
# Define a custom callback for capturing weights
class WeightCapture(Callback):
    "Capture the weights of each layer of the model"
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.weights = []
        self.epochs = []
 
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch) # remember the epoch axis
        weight = {}
        for layer in self.model.layers:
            if not layer.weights:
                continue
            name = layer.weights[0].name.split("/")[0]
            weight[name] = layer.weights[0].numpy()
        self.weights.append(weight)


# Define a custom callback for adaptive adjustment of learning rate
class LearningRate_Callback(Callback):
    def __init__(self, initial_lr, decay_factor):
        super(LearningRate_Callback, self).__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor

    def on_epoch_begin(self, epoch, logs=None):
        # Adjust the learning rate at the beginning of each epoch
        if (epoch > 0 and epoch <= 150):
            self.lr = self.initial_lr * np.exp(-self.decay_factor*epoch)  
            keras.backend.set_value(self.model.optimizer.lr, self.lr)
            print('\nLearning rate adjusted to {:.7f}\n'.format(float(self.model.optimizer.lr)))
        if epoch > 150:
            self.lr = self.initial_lr * np.exp(-self.decay_factor*150)  
            keras.backend.set_value(self.model.optimizer.lr, self.lr)
            print('\nLearning rate adjusted to {:.7f}\n'.format(float(self.model.optimizer.lr)))
##################



##################
# #### Compile & Fit the model 

# Create empty lists to store the history of training losses and component metrics
training_losses_history = []; val_losses_history = []
training_component_metrics_history = [[] for _ in range(len(loss_weights))]
val_component_metrics_history = [[] for _ in range(len(loss_weights))]
##################


##################
# Set callbacks
# Define the EarlyStopping callback
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=convergence_patience,
    min_delta=convergence_threshold,
    restore_best_weights=True)


# Instantiate the callback and pass it when compiling your model
lr_callback = LearningRate_Callback(initial_lr = lr_, decay_factor = decay_factor_)
##################



# Build keras model

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) `tensorflow` random seed
# 3) `python` random seed
keras.utils.set_random_seed(59298)

keras_model = build_keras_model(n_feat_atom, nfeat_bond, n_feat_global, 
                                activ, dropout_rate, energy_grid_size, n1, n2, n3, n4, n5, n6, n7)

# build graph_converter
graph_converter = build_graph_converter(r_cutoff, gaussian_width, nfeat_bond)

# Make a a graph_model, which is a composition of a keras model and a graph_converter
graph_model = GraphModel(keras_model, graph_converter)

# Instantiate the callback and pass it when compiling your model
weightcapture_callback = WeightCapture(graph_model.model)

# Prepapre data generators
train_generator, test_generator =  prepare_data_generators(graph_model, 
                                                           structures_train, 
                                                           targets_train, structures_test, targets_test, 
                                                           batch_size)

#------------------#
# Use the custom loss function with additional parameters
custom_loss_function = custom_loss_wrapper(energy_grid_size, energy_grid_delta, loss_weights)
#------------------#


# Define the optimizer
opt = keras.optimizers.AdamW()

# Compile the model
graph_model.compile(optimizer=opt, loss=custom_loss_function, 
                    metrics = [metric_1_wrapper(energy_grid_size, energy_grid_delta, loss_weights), 
                               metric_2_wrapper(energy_grid_size, energy_grid_delta, loss_weights)])
##################



##################
# Train the model for the current phase
history = graph_model.fit(
    train_generator,
    epochs=n_epochs,  # Adjust the number of epochs 
    verbose=2, 
    validation_data=test_generator,
    callbacks=[early_stopping_callback, weightcapture_callback, lr_callback],
)
##################


##################
# Append the training loss history 
training_losses_history.extend(history.history['loss'])
val_losses_history.extend(history.history['val_loss'])

# Append the component metrics to their respective lists 
for i in range(len(loss_weights)):
    component_metric = f"component_{i + 1}"
    training_component_metrics_history[i].extend(history.history[component_metric])
    
for i in range(len(loss_weights)):
    component_metric = f"val_component_{i + 1}"
    val_component_metrics_history[i].extend(history.history[component_metric]) 
##################
    

##################
# Calculate average over the last 20 epochs
num_epochs_to_average = 20

# Average component metrics
for i in range(len(loss_weights)):
    avg_training_component_metric = np.mean(training_component_metrics_history[i][-num_epochs_to_average:])
    avg_val_component_metric = np.mean(val_component_metrics_history[i][-num_epochs_to_average:])
    
    print(f"Average Training Component {i + 1} Metric over the Last {num_epochs_to_average} Epochs: {avg_training_component_metric}")
    print(f"Average Validation Component {i + 1} Metric over the Last {num_epochs_to_average} Epochs: {avg_val_component_metric}")
##################

    
##################
### Plot trial results 
# Set range 
k = 0
# Plot training and validation losses 
plt.figure(figsize=(12, 6))
plt.plot(training_losses_history[k:], label='Training Loss', alpha=0.7)
plt.plot(val_losses_history[k:], label='Validation Loss', alpha=0.7)
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_validation_loss.png')  # Save the figure

# Plot component metrics and their validation counterparts 
for i in range(len(loss_weights)):
    plt.figure(figsize=(12, 6))
    training_component_metric = f"component_{i + 1}_loss"
    val_component_metric = f"val_component_{i + 1}_loss"
    plt.plot(training_component_metrics_history[i][k:], label=training_component_metric, alpha=0.7)
    plt.plot(val_component_metrics_history[i][k:], label=val_component_metric, alpha=0.7)
    plt.title(f'Component {i + 1} Metric Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    component_filename = f'component_{i + 1}_metric.png'
    plt.savefig(component_filename)  # Save the figure
##################



##################
# Save model
graph_model.save_model('model.hd5')   
##################



print('---------------------------------------------')
print('---------------------------------------------')


##################
# Print layer weights info
print('Number of layers = ', len(list(weightcapture_callback.weights[0].keys())))
list(weightcapture_callback.weights[0].keys())


def plotweights(capture_cb, save_folder='weights_plots'):
    "Plot the mean absolute values of weights across epochs for each layer and save figures and weights"
    num_layers = len(capture_cb.weights[0])
    
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for layer_idx, key in enumerate(capture_cb.weights[0]):
        fig, ax = plt.subplots(1, 1, sharex=True, constrained_layout=True, figsize=(8, 4), dpi=200)
        ax.set_title(f"Mean Absolute Weight - {key}")
        mean_absolute_weights = [np.abs(w[key]).mean() for w in capture_cb.weights]
        ax.plot(capture_cb.epochs, mean_absolute_weights, label=key)
        ax.scatter(capture_cb.epochs, mean_absolute_weights, s=5)  # Add scatter points
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})  # Adjust legend placement and size

        # Save the figure
        fig_name = f"{key}_weights_plot.png"
        fig_path = os.path.join(save_folder, fig_name)
        plt.savefig(fig_path)
        plt.close()


plotweights(weightcapture_callback)
##################
