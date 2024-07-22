import streamlit as st
import torch
from torch.distributions.categorical import Categorical
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

RDLogger.DisableLog('rdApp.*')

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load your trained model
 ## Reading and processing text
with open('preprocessed_dataset.txt', 'r', encoding="utf8") as fp: smiles=fp.read()

smile_set = set(smiles)

import numpy as np

text = smiles
char_set = smile_set  # removes duplicates -> get unique characters

print(f"Text Length: {len(text)}")
print(f"Unique characters: {len(char_set)}")

chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted) # more efficient than dict

import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell
    


# Create RNN model
vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
model.load_state_dict(torch.load('finetunedmodel.pt', map_location= device))
model.to(device)

model.eval()

def sample(model, starting_str, max_length=300000, scale_factor=1.0):
    """
    starting_str: short starting string
    max_length: max length of generated text
    """
    
    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str  # initially set it equal to the input str

    hidden, cell = model.init_hidden(batch_size=1)
    
    # hidden = torch.zeros(1, 1, rnn_hidden_size)
    # cell = torch.zeros(1, 1, rnn_hidden_size)
    encoded_input = encoded_input.to(device)
    cell = cell.to(device)
    hidden = hidden.to(device)
    for c in range(len(starting_str) - 1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)
    last_char = encoded_input[:, -1]
    for i in tqdm(range(max_length)):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])



    return generated_str

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES string

    descriptors = {
        'SAScore': sascorer.calculateScore(mol),
        'QED': Chem.QED.default(mol), 
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        
    }
    return descriptors


# Streamlit app
st.title('New Compounds Generator')

start_string = st.text_input('Enter the starting string:', value='C')
scale_factor = st.slider('Scale Factor', min_value=0.1, max_value=5.0, value=1.0)
max_length = st.slider('Max Length', min_value=100, max_value=5000, value=3000)
num_smiles = st.slider('Number of SMILES to Generate', min_value=1, max_value=10, value=5)



if st.button('Generate'):
    st.write('Generating SMILES...')
    generated_smiles = sample(model, start_string, max_length=max_length, scale_factor=scale_factor)
    
    # Split generated SMILES into individual strings
    list_of_smiles = generated_smiles.split("\n")
    
    # Validate SMILES
    valid_smiles = []
    for smi in list_of_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_smiles.append(smi)

    accuracy = (len(valid_smiles) / len(list_of_smiles)) * 100 if list_of_smiles else 0
    st.write(f"Accuracy = {len(valid_smiles)}/{len(list_of_smiles)} = {accuracy:.2f}%")

    # Display results in a 3x3 grid
    st.subheader('Top Valid SMILES')
    columns = st.columns(3)
    for i, smiles in enumerate(valid_smiles[:num_smiles]):
        col = columns[i % 3]
        with col:
            st.write(f'Compound {i+1} \n Smiles: {smiles}')
            
            # Display descriptors
            descriptors = calculate_descriptors(smiles)
            if descriptors:
                st.write(descriptors)
            
            # Display molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.image(Draw.MolToImage(mol), use_column_width=True)