from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import torch


device = "mps" if torch.backends.mps.is_available() else "cpu"

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
# int2chr = {i:ch for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted) # more efficient than dict

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES string
    
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
    }
    return descriptors

def generate_top_smiles(model, start_string, scale_factor, top_n=5):
    smiles_list = []
    
    for _ in range(100):  # Generate more than needed to get top N unique ones
        smiles = sample(model, start_string, scale_factor)
        if smiles not in smiles_list:
            smiles_list.append(smiles)
            if len(smiles_list) >= top_n:
                break
    
    return smiles_list


def sample(model, starting_str, max_length=300000, scale_factor=1.0):
    """
    starting_str: short starting string
    max_length: max length of generated text
    """

    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str # initially set it equal to the input str

    model.eval()
    hidden, cell = model.init_hidden(batch_size=1)
    hidden = hidden.to(device)
    encoded_input = encoded_input.to(device)
    cell = cell.to(device)
    for c in range(len(starting_str)-1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)
    last_char = encoded_input[:, -1]
    for i in tqdm(range(max_length)):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])

        if i %1000 == 0:
            torch.mps.empty_cache()


    return generated_str