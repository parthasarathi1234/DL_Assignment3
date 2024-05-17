# DL_Assignment3

This repository contains the implementation of a Seq2Seq model for a transliteration system, with two variants both implemented using PyTorch
    1. Without attention mechanism : A simple Seq2Seq model that uses a basic encoder-decoder architecture
    2. with attention mechanism : A Seq2Seq model that enhances the basic architecture with an attention mechanism, allowing the decoder to focus on different parts of the input sequence during each decoding step.

Below is a detailed description of the code and its structure.

Code structure

## Requirements
To run the code, you need the following libraries:
- Python 3.7+
- torch
- pandas
- wandb
- argparse

'''
Install the required libraries using pip:
```

### GPU Integration
The code is optimized to use a GPU if one is available, ensuring faster computations and more efficient training.
pip install torch pandas wandb argparse
     
### Data set 
Different csv datasets (Train, Validation, and Test) are loaded and preprocessed for the model.

-wp, --wandb_project             : Specifies the project name for Weights & Biases.
-e, --epochs                     : Number of epochs to train the model.
-lr, --learning_rate             : Learning rate for the optimizer.
-b, --batch_size                 : Batch size for training.
-embd_dim, --char_embd_dim       : Dimension of character embeddings.
-hid_neur, --hidden_layer_neurons: Number of neurons in hidden layers.
-num_layers, --number_of_layers  : Number of layers in the encoder and decoder.
-cell, --cell_type               : Type of RNN cell (RNN, LSTM, GRU).
-do, --dropout                   : Dropout probability.
-opt, --optimizer                : Optimization algorithm (adam, nadam).
-train_path, --train_path        : (required) Path to the training data CSV file.
-test_path, --test_path          : (required) Path to the testing data CSV file.
-val_path, --val_path            : (required) Path to the validation data CSV file.


#### To train the model without attention:
The cs23m035_4_without_attention.py script uses parse_arguments from the parser library to execute the training.
It can be run with the command

python cs23m035_4_without_attention.py --<parameterName> <value>

## Example

python train_without_attention.py -wp my_project -e 20 -lr 0.001 -b 32 -embd_dim 256 -hid_neur 256 -num_layers 2 -cell LSTM -do 0.3 -opt adam -train_path data/train.csv -test_path data/test.csv -val_path data/val.csv

#### To train the model with attention:

In the AttentionRNN.ipynb notebook, wandb parameters are integrated directly, and sweeps are run to find the best hyperparameters.
You can run the notebook by integrating your wandb account using your activation key to visualize the results.

python cs23m035_with_attention.py --<parameterName> <value>

## Example

python train_with_cs23m035_with_attention.py -wp my_project -e 20 -lr 0.001 -b 32 -embd_dim 256 -hid_neur 256 -num_layers 2 -cell LSTM -do 0.3 -opt adam -train_path data/train.csv -test_path data/test.csv -val_path data/val.csv

Running the Training Scripts
To train a basic Seq2Seq model without attention:

Ensure you have prepared your dataset and saved it as train.csv, test.csv, and validation.csv.
Execute the training script using the command provided above, adjusting the parameters as needed.

This setup ensures a comprehensive and efficient way to train and evaluate the attention-based RNN model for transliteration. By leveraging GPU capabilities, wandb for tracking, and custom preprocessing and training functions, the model is designed to perform well on transliteration tasks.

