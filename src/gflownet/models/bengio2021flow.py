"""
This is code adapted from Bengio et al. (2021), 'Flow Network based
Generative Models for Non-Iterative Diverse Candidate Generation',
from
   https://github.com/GFNOrg/gflownet

In particular, this model class allows us to compare to the same
target proxy used in that paper (sEH binding affinity prediction).
"""

import gzip
import os
import pickle  # nosec
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Batch, Data
from torch_geometric.nn import NNConv, Set2Set
from torch_sparse import coalesce

NUM_ATOMIC_NUMBERS = 56  # Number of atoms used in the molecules (i.e. up to Ba)

# These are the fragments used in the original paper, each fragment is a tuple
# (SMILES string, attachment atom idx).
# The attachment atom idx is where bonds between fragments are legal.
FRAGMENTS: list[tuple[str, list[int]]] =[('c1nnc(N)o1',[0]),
 ('c1ccc([N+](=O)[O-])o1',[0]), 
 ("CO", [1, 0]),
 ("O=c1[nH]cnc2[nH]cnc12", [3, 6]),
 ("S", [0, 0]),
 ("C1CNCCN1", [2, 5]),
 ("c1cc[nH+]cc1", [3, 1]),
 ("c1ccccc1", [0, 2]),
 ("C1CCCCC1", [0, 2]),
 ("CC(C)C", [1, 2]),
 ("C1CCOCC1", [0, 2]),
 ("c1cn[nH]c1", [4, 0]),
 ("C1CCNC1", [2, 0]),
 ("c1cncnc1", [0, 1]),
 ("O=c1nc2[nH]c3ccccc3nc-2c(=O)[nH]1", [8, 4]),
 ("c1ccncc1", [1, 0]),
 ("O=c1nccc[nH]1", [6, 3]),
 ("O=c1cc[nH]c(=O)[nH]1", [2, 4]),
 ("C1CCOC1", [2, 4]),
 ("C1CCNCC1", [1, 0]),
 ('C[C@@H]C', [0, 1]),
 ('CC(O)CO', [0]),
 ('CCCS(=O)(=O)O', [0]),
 ('CCC', [0, 1, 2, 3]),
 ('C(=O)C(O)C(C)(C)CO', [0]),
 ('C[C@H]CO', [0, 1]),
 ('C(C)CCC(C)CCCC(C)CCCC(C)C', [0]),
 ('C(=O)[C@H]C(C)C', [0, 2]),
 ('C[C@@H][C@H][C@H][C@H][C@H]C', [0, 1, 3, 5, 7, 9, 11]),
 ('C(=O)CCC=O', [0, 3, 4, 5]),
 ('C(=O)[C@H]CCC', [0, 2, 6]),
 ('NC(=S)N', [0, 3]),
 ('C(=O)[C@@H]C', [0, 2, 4]),
 ('CC[C@H](O)CCCCCC', [0]),
 ('CC', [0, 1]),
 ('C(=O)CCCCCCCC=O', [0, 9]),
 ('CC=C', [0]),
 ('CCCCCCCCCCCCCCCC', [0]),
 ('C(=O)CCCCCCC(=O)N/N=C/[C@@H](O)[C@H](O)[C@H](O)CO', [0]),
 ('CCCCN', [0]),
 ('CCCC(=O)O', [0, 1]),
 ('[C@@H](C(=O)O)C(C)C', [0, 4]),
 ('C(=O)[C@@H](C)N', [0]),
 ('[C@H](CC(C)C)C(=O)O', [0]),
 ('C(=O)C', [0, 2]),
 ('OP(=O)(F)O', [0, 4]),
 ('CC(N)C(=O)O', [0]),
 ('C(=O)NNC[C@@H](O)C', [0, 7]),
 ('C(Br)C(Cl)(Cl)Br', [0]),
 ('C(=O)CC', [0, 2, 3, 4]),
 ('CCC(N)C(=O)O', [0]),
 ('NC(N)=S', [0]),
 ('C(=O)N(CCCl)N=O', [0]),
 ('O[Si](C#CCCCCCC)(C(C)C)C(C)C', [0]),
 ('CCCC(C)CC=O', [0]),
 ('CCCCCCCCCC', [0, 9]),
 ('C(=O)CCCCCCC(=O)NN', [0]),
 ('C(=O)CCC', [0, 4]),
 ('SS(=O)(=O)O', [0]),
 ('OS(N)(=O)=O', [0]),
 ('CC(O)C(O)C(O)C(O)CO', [0]),
 ('CC(CC)CCCC', [0]),
 ('C(=O)NNC[C@H](O)CC', [0]),
 ('C(C)(C)C', [0]),
 ('C(O)C(Cl)(Cl)Cl', [0]),
 ('NC=O', [0]),
 ('C(=O)C[C@H](O)C[C@H](O)CCCCC', [0]),
 ('C', [0]),
 ('CCCCCC(=O)O', [0]),
 ('CCC[C@@](C)(O)C=C', [0]),
 ('SS', [0, 1]),
 ('C(O)C(=O)O', [0]),
 ('S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', [0]),
 ('[C@@H](CCC(N)=O)C(=O)O', [0]),
 ('C(=O)CCCCCCCCCCCCC', [0]),
 ('C(=O)[C@H](Br)C(C)C', [0]),
 ('[N+](C)(C)C', [0]),
 ('OS(=O)(=O)[O-]', [0]),
 ('CCCC(N)C(=O)O', [0]),
 ('C[C@@H](N)C(=O)O', [0]),
 ('[C@@H]1CO1', [0]),
 ('CC(N)=O', [0]),
 ('C(=O)C[C@H](O)C[C@@H]CCCCC', [0, 6]),
 ('[C@H]1O[C@@H]1', [0, 2]),
 ('CCCCO', [0]),
 ('CCCCCC', [0, 5]),
 ('CC[C@H](C)CCC', [0, 6]),
 ('NC(=N)NC', [0]),
 ('NC(=N)NO', [0]),
 ('C[C@@H](O)C[C@@H](O)CCCCCCCCCCCC#C', [0]),
 ('CC(=O)[O-]', [0]),
 ('C(=O)N(CCCl)[N+][O-]', [0]),
 ('C(C)(C)CO', [0]),
 ('[C@@H](CC)CO', [0]),
 ('CCS(=O)(=O)[O-]', [0]),
 ('C1CC1', [0]),
 ('C[C@H](O)CC(=O)O', [0]),
 ('CCCCCCCC(=O)O', [0]),
 ('C(=O)[C@@H]CCC(N)=O', [0, 2]),
 ('CC(O)C(O)C(O)C(O)C(=O)[O-]', [0]),
 ('C(=O)CC(O)(CC=O)C(=O)O', [0, 6]),
 ('N', [0]),
 ('[C@H](CC(=O)O)C(=O)O', [0]),
 ('OC(O)P(=O)(O)CCCN', [0, 2]),
 ('CCCCCCCCCCCCCCCCCC', [0]),
 ('C(=O)[C@H](O)C(C)(C)CO', [0]),
 ('CC(C)=O', [0]),
 ('C(=O)CC(C)C', [0]),
 ('SC', [0]),
 ('CCCCC', [0, 3]),
 ('CCCCCCC', [0, 6]),
 ('CCCCN(O)C(C)=O', [0]),
 ('C(=O)[C@@H](O)C(C)(C)CO', [0]),
 ('CC(F)(F)F', [0]),
 ('C1CC1(C)C', [0, 1]),
 ('C(CCC(N)=O)C(=O)O', [0]),
 ('NC(C)=N', [0]),
 ('C(=O)[C@@H](C#N)C=O', [0, 5]),
 ('C=O', [0]),
 ('SP(=O)(O)O', [0]),
 ('C(=O)CCC(=O)N(O)CCCCC', [0, 12]),
 ('C(CS)C(=O)O', [0]),
 ('C(=O)C(=C)C', [0]),
 ('CCCC[C@H](N)C(=O)O', [0]),
 ('C(=O)CCC(N)C(=O)O', [0]),
 ('[C@H]1CO1', [0]),
 ('C(=O)CCC(=O)N(O)CCCCCN', [0]),
 ('CCCCCCCCC(N)=O', [0]),
 ('CC(O)CC(O)CCCCCCCCCCCC=C', [0]),
 ('N(C)C', [0]),
 ('CCCO', [0]),
 ('C(=O)C(C)S', [0]),
 ('C(=O)[C@@H]CCC', [0, 2, 6]),
 ('C(=O)C[C@@H](CC(=O)O)C(=O)O', [0]),
 ('C/C(=N/O)C(N)=O', [0]),
 ('[C@@H]1C[C@H]1C', [0]),
 ('ONC(=N)N', [0]),
 ('[C@@H](C(=O)O)[C@H](C)CC', [0]),
 ('NCP(=O)(O)O', [0]),
 ('CCC(O)(P(=O)([O-])O)P(=O)(O)O', [0]),
 ('NC', [0]),
 ('N/C(=N/C)NC', [0]),
 ('C[C@@H]C', [0, 1]),
 ('NC(=N)N', [0, 3]),
 ('C(C)=O', [0]),
 ('CC(O)CC(=O)O', [0]),
 ('SC(=N)N', [0]),
 ('C1(N)CC1', [0]),
 ('C(=O)C=C', [0]),
 ('S[N+][O-]', [0]),
 ('CC(=O)O', [0]),
 ('S[NH2+][O-]', [0]),
 ('O[PbH2]O', [0, 2]),
 ('C(=O)CCN', [0]),
 ('C[C@@H](O)CC(=O)O', [0]),
 ('CCCC[C@@H](N)C(=O)O', [0]),
 ('CCCCCCCCC(=O)[O-]', [0]),
 ('C(C)(C)CCC[C@H](C)CC', [0, 9]),
 ('C(=O)[C@@H]CC', [0, 2, 5]),
 ('CCCCCCCCCCCCCC', [0]),
 ('CC[C@H](N)C(=O)O', [0]),
 ('[C@@H](C=O)[C@@H](O)[C@H](O)[C@H](O)CO', [0]),
 ('CC[C@@H](N)C(=O)O', [0]),
 ('C(=O)C=[N+]=[N-]', [0]),
 ('C(=O)CCCCCCCCCCCCCCC', [0]),
 ('C(=O)C(N)CS', [0]),
 ('CCCl', [0]),
 ('SS(C)(=O)=O', [0]),
 ('[C@@H](CCC(=O)O)C(=O)O', [0]),
 ('C(=O)[C@@H]CC(C)C', [0, 2]),
 ('C(CC(C)C)C(=O)O', [0]),
 ('OCF', [0]),
 ('C(N)=O', [0]),
 ('C(=O)CCCCCCC(=O)N/N=C/[C@H](O)[C@H](O)[C@@H](O)[C@H](C)O', [0]),
 ('CCCCCCCCCCC(=O)O', [0]),
 ('C(=O)CCC(=O)CN', [0]),
 ('[C@@H]1C[C@@H]1', [0, 2]),
 ('CCC[C@@H](C)CCO', [0]),
 ('C(C)(O)CN', [0]),
 ('CCB(O)O', [0]),
 ('CCCCCC[C@H](C(=C)C(=O)O)C(=O)O', [0]),
 ('[C@@H](C)C(=O)O', [0]),
 ('C(=O)[C@@H]CC(=O)O', [0, 2]),
 ('CCC(=O)O', [0]),
 ('C(=O)[C@H](C)N', [0]),
 ('C(=O)CCC(N)C(=O)[O-]', [0]),
 ('[C@@H]1[C@H]C1(C)C', [0, 1]),
 ('C(=O)C(=C)O', [0]),
 ('OP(=O)(OC)OC', [0]),
 ('C[C@@H](C)O', [0]),
 ('C(C)C', [0]),
 ('O', [0]),
 ('CC#CC', [0]),
 ('CCCCCCCCC', [0]),
 ('C(O)CCC(=O)O', [0]),
 ('CCC[C@H]C(N)=O', [0, 3]),
 ('CCCS(=O)(=O)[O-]', [0]),
 ('[C@@H](CO)C(=O)O', [0]),
 ('C(C)CP(=O)(O)O', [0]),
 ('N(CP(=O)(O)O)CP(=O)(O)O', [0]),
 ('CCCC(C)=O', [0]),
 ('CCC[C@H](N)C(=O)O', [0]),
 ('[N+]#N', [0]),
 ('NC(N)=N[N+](=O)[O-]', [0]),
 ('CC(C)O', [0]),
 ('CC(O)CC(O)CCCCCCCCCCCC#C', [0]),
 ('CCCCCN(O)C(C)=O', [0]),
 ('CCCCCC[C@@H](C(=C)C(=O)O)C(=O)O', [0]),
 ('C(C)CCCC(C)C', [0]),
 ('C(=O)[C@@H](N)CC(C)C', [0]),
 ('C(=O)C=O', [0, 2]),
 ('C(=O)O', [0]),
 ('CC(=O)[C@@H](O)[C@H](O)[C@H](O)C', [0, 9]),
 ('SCS', [0, 2]),
 ('C(CC)CO', [0]),
 ('C(=O)CC(C=O)S(=O)(=O)[O-]', [0, 4]),
 ('C(CC(=O)O)C(=O)O', [0]),
 ('C(=O)CC[C@@H](N)C(=O)O', [0]),
 ('[C@H](O)[C@@H]O', [0, 2]),
 ('S[As](C)C', [0]),
 ('NC(=N)NC#N', [0]),
 ('SP(=O)(S)S', [0, 3, 5]),
 ('CC(=O)CC[C@H](N)C(=O)O', [0]),
 ('[C@H](C[C@H](O)C(C)(O)CCC)CC', [0, 9, 11]),
 ('N(C)C(=N)N', [0]),
 ('C(=O)CCCCCCC(=O)N/N=C/[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)C(=O)O', [0]),
 ('O[Cd]O', [0, 2]),
 ('CC(=O)[C@H](O)[C@@H](O)C', [0, 7]),
 ('CCC(=O)[O-]', [0]),
 ('C(=O)[C@@H]CCC(=O)O', [0, 2]),
 ('[C@H]([C@H](C)CCCC)[C@@H]C[C@@H](C)C[C@H](O)CCCC[C@@H](O)C[C@H](O)[C@H](C)N',
  [0, 7]),
 ('C(=O)[C@H](CO)[C@H](O)CCCC[C@H](C)CCC', [0, 14]),
 ('C(C)(C)CCCC(C)CC', [0, 9]),
 ('SP(=S)(OC)OC', [0]),
 ('OCO/N=[N+](\\[O-])N(CC)CC', [0]),
 ('OP(=O)(O)O', [0, 4]),
 ('C(=O)CCCCCCC(=O)N/N=C/[C@H](O)[C@@H](O)[C@H](O)CO', [0]),
 ('NC(=N)SC', [0]),
 ('C(=O)/C(C)=N/NC(C)=O', [0]),
 ('CCN', [0]),
 ('C(C)(C)CC', [0]),
 ('CCCCCCCCC(=O)O', [0]),
 ('C(=O)[C@@H]C(C)C', [0, 2]),
 ('c1cc1=O', [0, 1]),
 ('C(CC)[C@@H](C)[N+](=O)[O-]', [0]),
 ('CC(C)(C)CCC', [0, 3]),
 ('C[C@H]C', [0, 1]),
 ('[C@@H](CCN)C(N)=O', [0]),
 ('CCCC(C)(O)C=C', [0]),
 ('C[C@@H](O)[C@@H](O)[C@@H](O)[C@@H](O)C', [0, 9]),
 ('OP(=O)([O-])O', [0, 4]),
 ('C[C@@H](O)CN', [0]),
 ('C(=O)[C@H](CC(C)C)[C@H](O)C(=O)NO', [0]),
 ('C(=O)[C@@H](N)CCC', [0, 6]),
 ('[C@@H](C)P(=O)(O)O', [0]),
 ('[C@@H](CC(C)C)C(=O)O', [0]),
 ('C(=O)CC[C@H](N)C(=O)O', [0]),
 ('CC=O', [0]),
 ('OC=C', [0]),
 ('C(CCC(CC)CCCC)CC(C)C', [0]),
 ('CCC[C@@H](N)C(=O)O', [0]),
 ('[C@@H](CS)C(=O)O', [0]),
 ('[C@@H](CCCC[C@@H](C)CCC)[C@@H](CO)C(=O)O', [0, 8]),
 ('CC(O)CC(=O)[O-]', [0]),
 ('CCCCCCCCCC(=O)O', [0]),
 ('CCCN', [0]),
 ('C(=O)CCCCCCCC', [0, 9]),
 ('NC(=S)SSC(=S)N', [0, 8]),
 ('NC(=N)N(C)C', [0]),
 ('NCS(=O)(=O)O', [0]),
 ('CC(C)C', [0]),
 ('C[C@H](N)C(=O)O', [0]),
 ('CCCCC(=O)O', [0]),
 ('CCCCCCCC', [0]),
 ('C(=O)CCCCCCC=O', [0, 8]),
 ('C(=O)CC[C@H](C)N', [0]),
 ('CCC[C@H](N)C', [0, 5]),
 ('C[C@@H](O)C[C@@H](O)CCCCCCCCCCCC=C', [0]),
 ('S', [0]),
 ('CCO', [0]),
 ('ON=C(N)N', [0]),
 ('C(=O)/C(C)=N\\O', [0]),
 ('OC', [0]),
 ('CCCC', [0, 3]),
 ('C(=O)C(=O)O', [0]),
 ('C(C)CC[C@H](C)CCC[C@H](C)CCCC(C)C', [0]),
 ('[C@@H](CC(C)C)[C@@H](O)CC(=O)O', [0]),
 ('C(C(F)(F)F)C(F)(F)F', [0]),
 ('OP(=S)(O)SCSP(=S)(O)O', [0, 10, 3, 12]),
 ('C(=O)[C@H](N)C', [0, 4]),
 ('[C@@H](CC(=O)O)C(=O)O', [0]),
 ('C[C@@H](O)[C@H](O)[C@@H](O)[C@@H](O)CO', [0]),
 ('C(=O)CN', [0]),
 ('C[C@H]CC(=O)O', [0, 1]),
 ('C(=O)CCCC', [0, 5]),
 ('CCC[C@@H]C(N)=O', [0, 3]),
 ('CCCCCCCCCCCC', [0]),
 ('C(=O)CCS', [0, 2]),
 ('OS(C)(=O)=O', [0]),
 ('C(=O)CCCCCCCCCCCCCCCCC', [0]),
 ('C(CCC(=O)O)C(=O)O', [0]),
 ('CCS', [0]),
 ('C(=O)CCCCCCC', [0]),
 ('[C@H](CC)C(=O)O', [0]),
 ('C[C@@H](O)[C@@H](O)[C@@H](O)[C@@H](O)CO', [0]),
 ('CCCCCCO', [0]),
 ('C(=O)C[C@H](O)[C@@H]CC(C)C', [0, 5]),
 ('C(=O)[C@@H](N)CCC(=O)O', [0]),
 ('C(=O)[C@@H]C(C)(C)C', [0, 2])
]


class MPNNet(nn.Module):
    def __init__(
        self,
        num_feat=14,
        num_vec=3,
        dim=64,
        num_out_per_mol=1,
        num_out_per_stem=105,
        num_out_per_bond=1,
        num_conv_steps=12,
    ):
        super().__init__()
        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_ops = num_out_per_stem
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps
        self.dropout_rate = 0

        self.act = nn.LeakyReLU()

        net = nn.Sequential(nn.Linear(4, 128), self.act, nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr="mean")
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin3 = nn.Linear(dim * 2, num_out_per_mol)
        self.bond2out = nn.Sequential(
            nn.Linear(dim * 2, dim), self.act, nn.Linear(dim, dim), self.act, nn.Linear(dim, num_out_per_bond)
        )

    def forward(self, data, do_dropout=False):
        out = self.act(self.lin0(data.x))
        h = out.unsqueeze(0)
        h = F.dropout(h, training=do_dropout, p=self.dropout_rate)

        for i in range(self.num_conv_steps):
            m = self.act(self.conv(out, data.edge_index, data.edge_attr))
            m = F.dropout(m, training=do_dropout, p=self.dropout_rate)
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            h = F.dropout(h, training=do_dropout, p=self.dropout_rate)
            out = out.squeeze(0)

        global_out = self.set2set(out, data.batch)
        global_out = F.dropout(global_out, training=do_dropout, p=self.dropout_rate)
        per_mol_out = self.lin3(global_out)  # per mol scalar outputs
        return per_mol_out


def request():
    return requests.get(
        "https://github.com/GFNOrg/gflownet/raw/master/mols/data/pretrained_proxy/best_params.pkl.gz",
        stream=True,
        timeout=30,
    )


def download(location):
    f = request()
    location.parent.mkdir(exist_ok=True)
    with open(location, "wb") as fd:
        for chunk in f.iter_content(chunk_size=128):
            fd.write(chunk)


def load_weights(cache, location):
    if not cache:
        return pickle.load(gzip.open(request().raw))  # nosec

    try:
        gz = gzip.open(location)
    except gzip.BadGzipFile:
        download(location)
        gz = gzip.open(location)
    except FileNotFoundError:
        download(location)
        gz = gzip.open(location)
    return pickle.load(gz)  # nosec


def load_original_model(cache=True, location=Path(__file__).parent / "cache" / "bengio2021flow_proxy.pkl.gz"):
    num_feat = 14 + 1 + NUM_ATOMIC_NUMBERS
    mpnn = MPNNet(num_feat=num_feat, num_vec=0, dim=64, num_out_per_mol=1, num_out_per_stem=105, num_conv_steps=12)

    params = load_weights(cache, location)
    param_map = {
        "lin0.weight": params[0],
        "lin0.bias": params[1],
        "conv.bias": params[3],
        "conv.nn.0.weight": params[4],
        "conv.nn.0.bias": params[5],
        "conv.nn.2.weight": params[6],
        "conv.nn.2.bias": params[7],
        "conv.lin.weight": params[2],
        "gru.weight_ih_l0": params[8],
        "gru.weight_hh_l0": params[9],
        "gru.bias_ih_l0": params[10],
        "gru.bias_hh_l0": params[11],
        "set2set.lstm.weight_ih_l0": params[16],
        "set2set.lstm.weight_hh_l0": params[17],
        "set2set.lstm.bias_ih_l0": params[18],
        "set2set.lstm.bias_hh_l0": params[19],
        "lin3.weight": params[20],
        "lin3.bias": params[21],
    }
    for k, v in param_map.items():
        mpnn.get_parameter(k).data = torch.tensor(v)
    return mpnn


_mpnn_feat_cache = [None]


def mpnn_feat(mol, ifcoord=True, panda_fmt=False, one_hot_atom=False, donor_features=False):
    atomtypes = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    bondtypes = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, BT.UNSPECIFIED: 0}

    natm = len(mol.GetAtoms())
    ntypes = len(atomtypes)
    # featurize elements
    # columns are: ["type_idx" .. , "atomic_number", "acceptor", "donor",
    # "aromatic", "sp", "sp2", "sp3", "num_hs", [atomic_number_onehot] .. ])

    nfeat = ntypes + 1 + 8
    if one_hot_atom:
        nfeat += NUM_ATOMIC_NUMBERS
    atmfeat = np.zeros((natm, nfeat))

    # featurize
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx = atomtypes.get(atom.GetSymbol(), 5)
        atmfeat[i, type_idx] = 1
        if one_hot_atom:
            atmfeat[i, ntypes + 9 + atom.GetAtomicNum() - 1] = 1
        else:
            atmfeat[i, ntypes + 1] = (atom.GetAtomicNum() % 16) / 2.0
        atmfeat[i, ntypes + 4] = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        atmfeat[i, ntypes + 5] = hybridization == HybridizationType.SP
        atmfeat[i, ntypes + 6] = hybridization == HybridizationType.SP2
        atmfeat[i, ntypes + 7] = hybridization == HybridizationType.SP3
        atmfeat[i, ntypes + 8] = atom.GetTotalNumHs(includeNeighbors=True)

    # get donors and acceptors
    if donor_features:
        if _mpnn_feat_cache[0] is None:
            fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            _mpnn_feat_cache[0] = factory
        else:
            factory = _mpnn_feat_cache[0]
        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == "Donor":
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    atmfeat[k, ntypes + 3] = 1
            elif feats[j].GetFamily() == "Acceptor":
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    atmfeat[k, ntypes + 2] = 1
    # get coord
    if ifcoord:
        coord = np.asarray([mol.GetConformer(0).GetAtomPosition(j) for j in range(natm)])
    else:
        coord = None
    # get bonds and bond features
    bond = np.asarray([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
    bondfeat = [bondtypes[bond.GetBondType()] for bond in mol.GetBonds()]
    bondfeat = onehot(bondfeat, num_classes=len(bondtypes) - 1)

    return atmfeat, coord, bond, bondfeat


def mol_to_graph_backend(atmfeat, coord, bond, bondfeat, props={}, data_cls=Data):
    "convert to PyTorch geometric module"
    natm = atmfeat.shape[0]
    # transform to torch_geometric bond format; send edges both ways; sort bonds
    atmfeat = torch.tensor(atmfeat, dtype=torch.float32)
    if bond.shape[0] > 0:
        edge_index = torch.tensor(np.concatenate([bond.T, np.flipud(bond.T)], axis=1), dtype=torch.int64)
        edge_attr = torch.tensor(np.concatenate([bondfeat, bondfeat], axis=0), dtype=torch.float32)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, natm, natm)
    else:
        edge_index = torch.zeros((0, 2), dtype=torch.int64)
        edge_attr = torch.tensor(bondfeat, dtype=torch.float32)

    # make torch data
    if coord is not None:
        coord = torch.tensor(coord, dtype=torch.float32)
        data = data_cls(x=atmfeat, pos=coord, edge_index=edge_index, edge_attr=edge_attr, **props)
    else:
        data = data_cls(x=atmfeat, edge_index=edge_index, edge_attr=edge_attr, **props)
    return data


def onehot(arr, num_classes, dtype=np.int32):
    arr = np.asarray(arr, dtype=np.int32)
    assert len(arr.shape) == 1, "dims other than 1 not implemented"
    onehot_arr = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    onehot_arr[np.arange(arr.shape[0]), arr] = 1
    return onehot_arr


def mol2graph(mol, floatX=torch.float, bonds=False, nblocks=False):
    rdmol = mol
    if rdmol is None:
        g = Data(
            x=torch.zeros((1, 14 + NUM_ATOMIC_NUMBERS)),
            edge_attr=torch.zeros((0, 4)),
            edge_index=torch.zeros((0, 2)).long(),
        )
    else:
        atmfeat, _, bond, bondfeat = mpnn_feat(mol, ifcoord=False, one_hot_atom=True, donor_features=False)
        g = mol_to_graph_backend(atmfeat, None, bond, bondfeat)
    stem_mask = torch.zeros((g.x.shape[0], 1))
    g.x = torch.cat([g.x, stem_mask], 1).to(floatX)
    g.edge_attr = g.edge_attr.to(floatX)
    if g.edge_index.shape[0] == 0:
        g.edge_index = torch.zeros((2, 1)).long()
        g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).to(floatX)
    return g


def mols2batch(mols):
    batch = Batch.from_data_list(mols)
    return batch
