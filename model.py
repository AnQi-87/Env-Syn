import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(
    filename='Env-Syn.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    filemode='w'
)
logger = logging.getLogger(__name__)

class ECE(nn.Module):
    def __init__(self, hidden_dim, rank=None, drop=0.1):
        super().__init__()
        self.h = hidden_dim
        r = rank or max(16, hidden_dim // 8)

        in_dim = hidden_dim * 4 
        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU()
        )

        self.geglu_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.proj_1  = nn.Linear(hidden_dim, hidden_dim)

        self.U = nn.Linear(hidden_dim, r, bias=False)
        self.V = nn.Linear(hidden_dim, r, bias=False)
        self.Wb = nn.Linear(r, hidden_dim, bias=False)

        self.se = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(8, hidden_dim // 8)),
            nn.SiLU(),
            nn.Linear(max(8, hidden_dim // 8), hidden_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(drop)
        self.ln = nn.LayerNorm(hidden_dim)

    @staticmethod
    def geglu(x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.gelu(x2)

    def forward(self, cell_feat, other_drug_feat):
        c = cell_feat
        d = other_drug_feat

        z0 = torch.cat([c, d, c * d, c - d], dim=-1)    
        z  = self.pre(z0)                             

        z_gate = self.geglu(self.geglu_1(z))            
        z_gate = self.proj_1(self.dropout(z_gate))   

        bil = self.U(c) * self.V(d)               
        bil = self.Wb(self.dropout(bil))             

        out = z + z_gate + bil                    
        scale = self.se(out)                      
        out = out * scale

        return self.ln(out)

class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim, hidden=None):
        super().__init__()
        h = hidden or max(in_dim, cond_dim)
        self.gen = nn.Sequential(
            nn.Linear(cond_dim, h), nn.GELU(),
            nn.Linear(h, 2 * in_dim)  
        )
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, x, cond):
        gamma_beta = self.gen(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)    
        return self.ln(x + gamma * x + beta)

class CellMLPEncoder(nn.Module):
    def __init__(self, cell_dim=954, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        in_dim = cell_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim * 2))
            layers.append(nn.LayerNorm(hidden_dim * 2))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim * 2
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, cell_vec):
        return self.mlp(cell_vec)

class DrugFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    def forward(self, drug1_feat, drug2_feat):
        concat_feat = torch.cat([drug1_feat, drug2_feat], dim=-1)
        gate = self.gate(concat_feat)
        transformed = self.transform(concat_feat)
        interaction = gate * transformed + (1-gate) * torch.abs(drug1_feat - drug2_feat)
        return interaction


###############################################################################
class EnvSyn(nn.Module):
    def __init__(self, num_node_features=78, mol_dim=768, hidden_dim=128,
                 cell_dim=954, task='cls', semantic_tokens=8):
        super().__init__()
        assert task in ['cls', 'reg']
        self.task = task
        self.hidden_dim = hidden_dim
        self.adaptive = DrugFusion(hidden_dim=hidden_dim)
        self.cell_encoder = CellMLPEncoder(cell_dim=cell_dim, hidden_dim=hidden_dim)
        self.cond_merge = ECE(hidden_dim)

        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.film_embed = FiLM(in_dim=hidden_dim, cond_dim=hidden_dim)

        in_dim = hidden_dim*4 
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.out_layer = nn.Linear(hidden_dim, 1) 


    def forward(self, data1, data2, sup_labels=None):
        drug1, drug2 = data1.mol, data2.mol
        d1_b, d2_b = self.mol_proj(drug1), self.mol_proj(drug2)

        cell_vec = data1.cell
        cell_feat = self.cell_encoder(cell_vec)               

        cond1 = self.cond_merge(cell_feat, d2_b)  
        cond2 = self.cond_merge(cell_feat, d1_b)  

        d1_bi = self.film_embed(d1_b, cond1)    
        d2_bi = self.film_embed(d2_b, cond2)    

        interaction1 = self.adaptive(d1_b,   d2_b)
        interaction2 = self.adaptive(d1_bi,  d2_bi)

        combined = torch.cat([d1_bi, d2_bi, interaction1, interaction2], dim=-1)
        h = self.head(combined)
        out = self.out_layer(h).squeeze(-1)
        prob = torch.sigmoid(out)
        return prob
  