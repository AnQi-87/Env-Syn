import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
import random
import copy
import logging
from tqdm import tqdm

logging.basicConfig(
    filename='Env-Syn.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    filemode='w'
)
logger = logging.getLogger(__name__)

class TestbedDataset(InMemoryDataset):   
    def __init__(self, root='data', dataset=None,
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None,  mol_dict=None, mol_dim=768, drug2id=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        self.mol_dict = mol_dict if mol_dict is not None else {}
        self.mol_dim = mol_dim
        self.drug2id = drug2id

        logger.info('Start processing data...')

        if os.path.isfile(self.processed_paths[0]):
            logger.info('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            logger.info('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None): 
            if cellId in row[0]:
                return row[1:]  
        return False
    
    def _canonicalize_smiles(self, smiles):
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return smiles

    def _get_mol_vec(self, smile):
        """Get precomputed MoLFormer features"""
        canon_smile = self._canonicalize_smiles(smile)
        
        if canon_smile in self.mol_dict:
            v = np.asarray(self.mol_dict[canon_smile], dtype=np.float32)
            return v
        elif smile in self.mol_dict:
            v = np.asarray(self.mol_dict[smile], dtype=np.float32)
            return v
        
        return np.zeros((self.mol_dim,), dtype=np.float32)

    def process(self, xd, xt, xt_featrue, y):  
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        logger.info(f'Number of data points: {data_len}')

        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            cell = self.get_cell_feature(target, xt_featrue)  
            if cell is False:  
                logger.info('cell', cell)
                sys.exit()
            new_cell = [float(n) for n in cell]

            mol_vec = self._get_mol_vec(smiles)

            drug_id = self.drug2id[smiles]

            GCNData = DATA.Data(                
                y=torch.tensor([labels], dtype=torch.float32)
            )
            GCNData.drug_id = torch.tensor([drug_id], dtype=torch.long)
            GCNData.cell = torch.tensor([new_cell], dtype=torch.float32)
            GCNData.mol = torch.tensor([mol_vec], dtype=torch.float32)
            
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logger.info('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        if isinstance(AUCs, (list, tuple)):
            f.write('\t'.join(map(str, AUCs)) + '\n')
        else:
            f.write(str(AUCs) + '\n')