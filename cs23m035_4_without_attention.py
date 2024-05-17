import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import copy
from torch.utils.data import Dataset, DataLoader
import gc
import random
import wandb

def padding_to_string(source_data, max_length, length): # return string with padding
    strings = []
    for i in range(length):
        str_padding = START + source_data[i]   # adding start token and end token
        str_padding = str_padding + END
        str_padding = str_padding[:max_length]       #  
        padding_length = max_length - len(str_padding)
        if(padding_length>0):
            str_padding = str_padding + PADDING * padding_length        
        strings.append(str_padding)
    return strings


def generate_string_to_sequence(source_data, char_to_index, length):  # char to index converting
    source_sequences = []
    for i in range(length):
        index=[]
        for j in source_data[i]:
            index.append(char_to_index[j])
        source_sequences.append(torch.tensor(index, device=device))
    source_sequences = pad_sequence(source_sequences, batch_first=True, padding_value=2)
    return source_sequences


def character_adding(c, data, source_char_index, source_chars, source_index_char):  # characters adding to the data structure
    if data[source_char_index].get(c) is None:
        data[source_chars].append(c)              
        idx = len(data[source_chars]) - 1
        data[source_char_index][c] = idx
        data[source_index_char][idx] = c

def preprocess_data(source_data, target_data, length_data):   # preprocessing the source data and target data
    data = {
        "source_chars": [START, END, PADDING],
        "target_chars": [START, END, PADDING],
        "source_char_index": {START: 0, END:1, PADDING:2},
        "source_index_char": {0:START, 1: END, 2:PADDING},
        "target_char_index": {START: 0, END:1, PADDING:2},
        "target_index_char": {0:START, 1: END, 2:PADDING},
        "source_len": 3,
        "target_len": 3,
        "source_data": source_data,
        "target_data": target_data,
        "source_data_seq": [],
        "target_data_seq": []
    }
    
    
    x_max_length=0
    y_max_length=0
    for i in range(len(source_data)):   # finding maximum length input and output
        x_max_length=max(x_max_length,len(source_data[i]))
        y_max_length=max(y_max_length,len(target_data[i]))
        
        
    data["INPUT_MAX_LENGTH"] = x_max_length + 2   # adding 2 for start and end
    data["OUTPUT_MAX_LENGTH"] = y_max_length + 2
   
    
    padded_source_strings = padding_to_string(source_data, data["INPUT_MAX_LENGTH"], length_data)    # adding start token and end token and padding
    padded_target_strings = padding_to_string(target_data, data["OUTPUT_MAX_LENGTH"], length_data)
    
    for i in range(length_data):   # for every string 
        for c in padded_source_strings[i]:        # for every character in string 
            character_adding(c, data, "source_char_index", "source_chars", "source_index_char")
        for c in padded_target_strings[i]:
            character_adding(c, data, "target_char_index", "target_chars", "target_index_char")

    data['source_data_seq'] = generate_string_to_sequence(padded_source_strings,  data['source_char_index'], len(padded_source_strings))
    data['target_data_seq'] = generate_string_to_sequence(padded_target_strings,  data['target_char_index'], len(padded_target_strings))
    
    return data
def get_cell_type(state):  # selecting based on the cell
    if(state == "RNN"):
        return nn.RNN
    else:
        if(state=="GRU"):
            return nn.GRU
        elif(state=="LSTM"):
            return nn.LSTM
        
class Encoder(nn.Module):
    def __init__(self, h_params, data, device, p_dropout, p_cell, p_hidden_layers, p_no_of_layers, p_char_dim, length):
        super(Encoder, self).__init__()
        self.device=device
        self.h_params = h_params
        self.embedding = nn.Embedding(length, p_char_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.cell = get_cell_type(p_cell)(p_char_dim, p_hidden_layers, num_layers=p_no_of_layers, dropout= p_dropout, batch_first=True)
        
        
    def forward(self, current_input, prev_state):  # returning state and output
        h, prev_state = self.cell(self.dropout(self.embedding(current_input)), prev_state)
        return h, prev_state

    def getInitialState(self, no_of_layers, batchsize, hidden_layers):
        return torch.zeros(no_of_layers, batchsize, hidden_layers, device=self.device)

    
class Decoder(nn.Module):
    def __init__(self, h_params, data,device, p_dropout, p_cell, p_hidden_layers, p_no_of_layers, p_char_dim, length):
        super(Decoder, self).__init__()
        self.h_params = h_params
        self.cell = get_cell_type(p_cell)(p_char_dim, p_hidden_layers,num_layers=p_no_of_layers,dropout= p_dropout, batch_first=True)
        self.embedding = nn.Embedding(length, p_char_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(p_hidden_layers, length)
        self.softmax = nn.LogSoftmax(dim=2)
        

    def forward(self, current_input, prev_state):
        curr_embd = F.relu(self.embedding(current_input))
        curr_embd = self.dropout(curr_embd)
        output, prev_state = self.cell(curr_embd, prev_state)
        output = self.softmax(self.fc(output))
        return output, prev_state


def decoder_training(h_params, source_sequence, encoder, data):
    encoder_hidden = encoder.getInitialState(h_params["number_of_layers"], h_params["batch_size"],h_params["hidden_layer_neurons"])
    if h_params["cell_type"] == "LSTM":  # checking cell is lstm or not
        encoder_hidden = (encoder_hidden, encoder.getInitialState(h_params["number_of_layers"], h_params["batch_size"],h_params["hidden_layer_neurons"]))
    encoder_outputs, encoder_hidden = encoder(source_sequence, encoder_hidden)
    return torch.full((h_params["batch_size"], 1), data['target_char_index'][START], device=device), encoder_hidden
 
def loss_calculation(correct_predictions, total_predictions, total_loss, number_of_batches):  # calculating loss and accuracy
    accu=correct_predictions/total_predictions
    loss=total_loss/number_of_batches
    return accu, loss

def evaluate(encoder, decoder, data, dataloader, device, h_params, loss_fn, max_length_y, total_pre, batch_total):   # validating validation accuracy
    correct_predictions = 0
    total_loss = 0
    total_predictions = total_pre
    number_of_batches = batch_total
    for batch_num, (source_sequence, target_sequence) in enumerate(dataloader):
        input_tensor = source_sequence
        target_tensor = target_sequence
        encoder.eval()
        decoder.eval()

        loss = 0
        correct = 0
                
        with torch.no_grad():
            decoder_input_tensor, decoder_hidden = decoder_training(h_params, source_sequence, encoder, data)
            decoder_actual_output = []
            for di in range(max_length_y):
                curr_target_chars = target_tensor[:, di]
                decoder_output, decoder_hidden = decoder(decoder_input_tensor, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input_tensor = topi.squeeze().detach()
                decoder_actual_output.append(decoder_input_tensor)
                decoder_input_tensor = decoder_input_tensor.view(h_params["batch_size"], 1)
                            
                decoder_output = decoder_output[:, -1, :]
                loss+=(loss_fn(decoder_output, curr_target_chars))

            decoder_actual_output = torch.cat(decoder_actual_output,dim=0).view(max_length_y, h_params["batch_size"]).transpose(0,1)
            correct = (decoder_actual_output == target_tensor).all(dim=1).sum().item()
            loss_prediction=loss.item()/max_length_y

        correct_predictions+=correct
        total_loss +=loss
    
    accuracy, total_loss = loss_calculation(correct_predictions, total_predictions, total_loss, number_of_batches)
    return accuracy, total_loss



def optimizer_fun(h_params, encoder, decoder):  # encoder optimizer and decoder optimizer and getting based on adam and nadam
    x=[]  
    y=[]
    if(h_params["optimizer"]=="adam"):
        x=optim.Adam(encoder.parameters(), lr=h_params["learning_rate"])
        y=optim.Adam(decoder.parameters(), lr=h_params["learning_rate"])
    elif(h_params["optimizer"]=="nadam"):
        x=optim.NAdam(encoder.parameters(), lr=h_params["learning_rate"])
        y=optim.NAdam(decoder.parameters(), lr=h_params["learning_rate"])
    return x,y
        

def train_loop(encoder, decoder,h_params, data, data_loader, val_dataloader, device, y_max_length, no_of_epochs, batchsize, cell, total_pre, batch_total):   #  loop the train data 

    encoder_optimizer, decoder_optimizer = optimizer_fun(h_params, encoder, decoder)
    total_predictions = total_pre
    total_batches = batch_total
    loss_fn = nn.NLLLoss()
    for i in range(no_of_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        total_correct = 0
        for batch_num, (source_batch, target_batch) in enumerate(data_loader):
            loss = 0
            correct = 0
            # Encoder
            encoder_initial_state = encoder.getInitialState(h_params["number_of_layers"], h_params["batch_size"],h_params["hidden_layer_neurons"])
            if(cell== "LSTM"):
                encoder_initial_state = (encoder_initial_state, encoder.getInitialState(h_params["number_of_layers"], h_params["batch_size"],h_params["hidden_layer_neurons"]))
            
            encoder_output, encoder_current_state = encoder(source_batch, encoder_initial_state)
            decoder_curr_state = encoder_current_state
            output_seq_len = y_max_length
            decoder_actual_output = []
            
            ran_num=random.random()
            # Decoder
            for i in range(y_max_length):
            
                if(i == 0):
                    decoder_input_tensor = target_batch[:, 0].view(batchsize,1)
                    
                curr_target_chars = target_batch[:, i]
                decoder_output, decoder_curr_state = decoder(decoder_input_tensor,decoder_curr_state)
                topv, topi = decoder_output.topk(1)

                decoder_input_tensor = topi.squeeze().detach()
                decoder_actual_output.append(decoder_input_tensor)

                if(i<output_seq_len-1):
                    if(ran_num<TEACHER_FORCING_RATIO):  # if it true then we giving truth output
                        decoder_input_tensor = target_batch[:, i+1].view(batchsize, 1)
                    else:                               # in this case we giving the previous predicted output
                        decoder_input_tensor = decoder_input_tensor.view(batchsize, 1)
                    
                decoder_output = decoder_output[:, -1, :]
                loss+=(loss_fn(decoder_output, curr_target_chars))

            decoder_actual_output = torch.cat(decoder_actual_output,dim=0).view(output_seq_len, batchsize).transpose(0,1)
            
            correct = (decoder_actual_output == target_batch).all(dim=1).sum().item()
            total_correct+=correct
            total_loss += loss.item()/output_seq_len
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        train_acc, train_loss = loss_calculation(total_correct, total_predictions, total_loss, total_batches)    
        val_acc, val_loss = evaluate(encoder, decoder, data, val_dataloader,device, h_params, loss_fn, data["OUTPUT_MAX_LENGTH"], len(val_dataloader.dataset), len(val_dataloader))

        print("epoch: ", i, " train acc:", train_acc, " train loss:", train_loss, " val acc:", val_acc, " val loss:", val_loss)
        wandb.log({"train_accuracy":train_acc, "train_loss":train_loss, "val_accuracy":val_acc, "val_loss":val_loss, "epoch":i})

    return encoder, decoder, loss_fn

class MyDataset(Dataset):
    def __init__(self, data):
        self.source_data_seq = data[0]
        self.target_data_seq = data[1]
    
    def __len__(self):
        return len(self.source_data_seq)
    
    def __getitem__(self, idx):
        source_data = self.source_data_seq[idx]
        target_data = self.target_data_seq[idx]
        return source_data, target_data

def train_val_data(x, y, h_params):
    data_set=[x,y]
    train_val_set=MyDataset(data_set)
    return DataLoader(train_val_set, batch_size=h_params["batch_size"], shuffle=True)

def processing_dataset(train_source, train_target, val_source, val_target, h_params):   # preocessing the data
    data = preprocess_data(copy.copy(train_source), copy.copy(train_target), len(train_source))

    data["source_len"] = len(data["source_chars"])
    data["target_len"] = len(data["target_chars"])

    train_dataloader = train_val_data(data["source_data_seq"], data['target_data_seq'], h_params)

    val_padded_source_strings=padding_to_string(val_source, data["INPUT_MAX_LENGTH"], len(val_source))
    val_padded_target_strings = padding_to_string(val_target, data["OUTPUT_MAX_LENGTH"], len(val_source))
    
    val_source_sequences = generate_string_to_sequence(val_padded_source_strings, data['source_char_index'], len(val_padded_source_strings))
    val_target_sequences = generate_string_to_sequence(val_padded_target_strings, data['target_char_index'], len(val_padded_target_strings))

    val_dataloader = train_val_data(val_source_sequences, val_target_sequences, h_params)
    return train_dataloader, val_dataloader, data

def training_function(h_params, device, train_x, train_y, val_x, val_y):
    train_dataloader, val_dataloader, data = processing_dataset(train_x, train_y, val_x, val_y, h_params)
    encoder = Encoder(h_params, data, device, h_params["dropout"], h_params["cell_type"], h_params["hidden_layer_neurons"], h_params["number_of_layers"], h_params["char_embd_dim"], data["source_len"]).to(device)
    decoder = Decoder(h_params, data, device, h_params["dropout"], h_params["cell_type"], h_params["hidden_layer_neurons"], h_params["number_of_layers"], h_params["char_embd_dim"], data["target_len"]).to(device)
    encoder,  decoder, loss_fn = train_loop(encoder, decoder,h_params, data, train_dataloader,val_dataloader,device, data["OUTPUT_MAX_LENGTH"], h_params["epochs"], h_params["batch_size"], h_params["cell_type"], len(train_dataloader.dataset), len(train_dataloader))
    return encoder, decoder, loss_fn


def parse_arguments():
    parser = argparse.ArgumentParser(description="train the model with hyperparameters")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DL_project_A3", help="project name used to track in the Weights & Biases dashboard")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="no of epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-embd_dim", "--char_embd_dim", type=int, default=128, help="Dimension of character embeddings")
    parser.add_argument("-hid_neur", "--hidden_layer_neurons", type=int, default=512, help="Number of neurons in hidden layers")
    parser.add_argument("-num_layers", "--number_of_layers", type=int, default=2, help="Number of layers in the encoder and decoder")
    parser.add_argument("-cell", "--cell_type", choices=["RNN", "LSTM", "GRU"], default="LSTM", help="RNN, LSTM, GRU")
    parser.add_argument("-do", "--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("-opt", "--optimizer", choices=["adam", "nadam"], default="adam", help="adam, nadam")
    parser.add_argument("-train_path", "--train_path", type=str, required=True, help="training data csv file path")
    parser.add_argument("-test_path", "--test_path", type=str, required=True, help=" testing data csv file path")
    parser.add_argument("-val_path", "--val_path", type=str, required=True, help=" validation data csv file path")

    args = parser.parse_args()

    return args


def main():
    wandb.login()
    args = parse_arguments()

    h_params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "char_embd_dim": args.char_embd_dim,
        "hidden_layer_neurons": args.hidden_layer_neurons,
        "number_of_layers": args.number_of_layers,
        "cell_type": args.cell_type,
        "dropout": args.dropout,
        "optimizer": args.optimizer
    }


    train_csv = args.train_path
    test_csv = args.test_path
    val_csv = args.val_path
    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)
    val_df = pd.read_csv(val_csv, header=None)
    train_x, train_y = train_df[0].to_numpy(), train_df[1].to_numpy()
    val_x, val_y = val_df[0].to_numpy(), val_df[1].to_numpy()

    config = h_params
    run = wandb.init(project=args.wandb_project, name=f"{config['cell_type']}_{config['optimizer']}_ep_{config['epochs']}_lr_{config['learning_rate']}_embd_{config['char_embd_dim']}_hid_lyr_neur_{config['hidden_layer_neurons']}_bs_{config['batch_size']}_enc_layers_{config['number_of_layers']}_dec_layers_{config['number_of_layers']}_dropout_{config['dropout']}", config=config)
    training_function(config, device, train_x, train_y, val_x, val_y)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
END = '>'
START = '<'
PADDING = '_'
TEACHER_FORCING_RATIO = 0.5

main()