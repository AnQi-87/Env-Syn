import csv
from itertools import islice
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
import tqdm
from utils_test import *
from torch_geometric.utils import degree, to_undirected
import logging
import torch
from rdkit.Chem.rdchem import BondType
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    filename='Env-Syn.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    filemode='w'
)
logger = logging.getLogger(__name__)

class MoLFormerProcessor:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct",trust_remote_code=True)
            self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct",trust_remote_code=True)
            self.model.eval()
            self.use_mol = True
            logger.info("MoLFormer successfully loaded")
        except Exception as e:
            logger.warning(f"MoLFormer loading failed: {str(e)}")
            self.use_mol = False

    def _canonicalize_smiles(self, smiles):
        """Use RDKit to normalize the SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Unable to resolve SMILES: {smiles}")
                return smiles  
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.warning(f"Normalization of SMILES failed: {smiles}, Error: {e}")
            return smiles
    
    def get_mol_features(self, smiles):
        """Get MolFormer features for a SMILES string - CORRECTED VERSION"""
        if not self.use_mol:
            return np.zeros(768)
            
        try:
            canon_smiles = self._canonicalize_smiles(smiles)
            inputs = self.tokenizer(canon_smiles, return_tensors="pt", padding=True, truncation=True, max_length=202)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state 
            attention_mask = inputs['attention_mask'] 
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            features = (sum_embeddings / sum_mask).cpu().numpy()[0]
            
            return features
        except Exception as e:
            logger.warning(f"MolFormer processing failed for SMILES: {smiles}, using zero vector. Error: {str(e)}")
            return np.zeros(768)

def save_mol_features(smiles_list, output_file):
    """Precompute and save MoLFormer features for all SMILES"""
    processor = MoLFormerProcessor()
    mol_dict = {}
    
    for smile in tqdm(smiles_list, desc="Processing MoLFormer features"):
        canon_smile = processor._canonicalize_smiles(smile)
        mol_dict[canon_smile] = processor.get_mol_features(smile)
    
    np.save(output_file, mol_dict)
    logger.info(f"Saved MoLFormer features to {output_file}")
    return mol_dict

def load_MoLFormer_embeddings(MoLFormer_file=None):
    if MoLFormer_file is None or not os.path.exists(MoLFormer_file):
        logger.info('mol is None')
        return {}
    
    try:
        logger.info('Loading mol...')
        mol_dict = np.load(MoLFormer_file, allow_pickle=True).item()
        return {k: np.asarray(v, dtype=np.float32) for k, v in mol_dict.items()}
    except Exception as e:
        logger.warning(f'Failed to load MoLFormer embeddings: {e}')
        return {}

def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set)) 


def creat_data(datafile, drug_smiles_file, cellfile, MoLFormer_file=None, precompute_mol=True):
    processor = MoLFormerProcessor()
    cell_features = []
    with open(cellfile) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    df_smi = pd.read_csv(drug_smiles_file)
    
    raw_smiles_list = list(df_smi['smile'])
    compound_iso_smiles = []
    for smile in raw_smiles_list:
        canon_smile = processor._canonicalize_smiles(smile)
        compound_iso_smiles.append(canon_smile)
    
    compound_iso_smiles = set(compound_iso_smiles)
    if precompute_mol and (MoLFormer_file is None or not os.path.exists(MoLFormer_file)):
       mol_dict = save_mol_features(compound_iso_smiles, MoLFormer_file)
    else:
        mol_dict = load_MoLFormer_embeddings(MoLFormer_file)
    

    df = pd.read_csv(datafile)
    
    drug1_raw, drug2_raw = list(df['drug1']), list(df['drug2'])
    drug1, drug2 = [], []
    
    for d1, d2 in zip(drug1_raw, drug2_raw):
        drug1.append(processor._canonicalize_smiles(d1))
        drug2.append(processor._canonicalize_smiles(d2))
    
    cell, label = list(df['cell']), list(df['label'])
    drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)
    
    all_task_smiles = set(drug1) | set(drug2)
    missing_smiles = all_task_smiles - set(compound_iso_smiles)
    if missing_smiles:
        logger.warning(f"The task contains {len(missing_smiles)} SMILES that are not in the drug dictionary, which may indicate a problem.")
    
    all_drugs = sorted(list(set(drug1) | set(drug2)))
    drug2id = {drug: idx for idx, drug in enumerate(all_drugs)}

    return drug1, drug2, cell, label, cell_features, mol_dict, drug2id

if __name__ == "__main__":
    cellfile = 'data/cell_features_954.csv'
    drug_smiles_file = 'data/smiles.csv'
    datafile = 'data/new_labels_0_10.csv'
    creat_data(datafile, drug_smiles_file, cellfile, MoLFormer_file="data/processed/MoLFormer.npy", precompute_mol=True)