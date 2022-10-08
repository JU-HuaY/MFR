# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:06:40 2020

@author: 华阳
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import defaultdict
from rdkit.Chem.rdchem import BondType
import pickle
import torch
# from SEC_pre import pre_ss
from graph_features import atom_features
from rdkit.Chem import AllChem, rdMolDescriptors
# from gensim.models import word2vec
from collections import defaultdict
# from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec

#model = word2vec.Word2Vec.load('model_300dim.pkl')
# rdkit GetBondType() result -> int
BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)

dicts = {"H":"A","R":"A","K":"A",
         "D":"B","E":"B","N":"B","Q":"B",
         "C":"C","X":"C",
         "S":"D","T":"D","P":"D","A":"D","G":"D","U":"D",
         "M":"E","I":"E","L":"E","V":"E",
         "F":"F","Y":"F","W":"F"}

protein_dict = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25 }

num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return node_features, adjacency

seq_rdic2 = ['A','B','C','D','E','F']
seq_dic2 = {w: i+1 for i,w in enumerate(seq_rdic2)}


def split_sequence(seq):
    protein = np.zeros(1200)
    for i in range(len(seq)):
        protein[i] = int(seq[i])
    return protein

def split_sequence2(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def first_sequence(sequence):
    # sequence = '-' + sequence + '='
    words = [protein_dict[sequence[i]]
             for i in range(len(sequence))]
    return np.array(words)

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}


# responses_label = np.load("data/kiba_label.npy")
# responses_protein = open("data/kiba_label_key.txt", 'r').read().split("\n")


def zhu(dataset,fileinput):
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    compounds, compound_yuyi, adjacencies,morgan,proteins,interactions,ss_s = [], [], [], [], [],[],[]
    labels = []
    # proteins_f = []
    file_ = 'out/' + "O00141.txt"
    f = open(file_, "w")
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        drug_id, protein_id, smiles, sequences, interaction = data.strip().split(" ")
        if protein_id == "O00141" and float(interaction) > 11.5:
            f.write("drug:" + str(drug_id) + "  smiles:" +  str(smiles) + "  interaction:" +str(interaction) + "\n")
        # print(drug_id)
        # yuyi_vec = []
        # for i in range(len(smiles)):
        #     yuyi_vec.append(CHAR_SMI_SET[smiles[i]])
        # yuyi_vec = np.array(yuyi_vec)
        # compound_yuyi.append(yuyi_vec)
        # print(smiles)
        # print(sequences)
        # print(interaction)

        # atom_feature, adj = smile_to_graph(smiles)
        # print(atom_feature.shape)
        # mol = Chem.MolFromSmiles(smiles)
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        # npfp = np.array(list(fp.ToBitString())).astype('int8')
        # morgan.append(npfp)
        # print(npfp)
        # compounds.append(atom_feature)
        # # print(atom_feature.shape)
        # adjacencies.append(adj)
        # newsequence = ''
        # for i in range(len(sequences)):
        #     newsequence += dicts[sequences[i]]
        # newsequence2 = ''
        # for i in range(len(newsequence)):
        #     newsequence2 += str(seq_dic2[newsequence[i]])

        # protein_fea = buling(newsequence2)
        # protein_feature = split_sequence2(newsequence2, 3)
        # protein_fea = split_sequence2(sequences, 3)
        # proteins_f.append(protein_feature)
        # ss_vector = pre_ss(sequences)
        # ss_s.append(ss_vector)

        #words = split_sequence(sequence, 3)
        # newsequence = ''
        #
        # for i in range(len(sequences)):
        #     newsequence += dicts[sequences[i]]
        # newsequence2 = ''
        # for i in range(len(newsequence)):
        #     newsequence2 += str(seq_dic2[newsequence[i]])
        #
        # #protein_fea = buling(newsequence2)
        # protein_feature = split_sequence2(newsequence2,3)
        # proteins.append(protein_feature)

        # interactions.append(np.array([float(interaction)]))
        # label_key = sequences[0:10]
        # # t = 0
        # for i in range(len(responses_protein)):
        #     if responses_protein[i] == label_key:
        #         # print(responses_label[i])
        #         labels.append(np.array(responses_label[i]))
        #         break
        # print(t)

    # responses_labels_ = np.array(labels)
    # print(responses_labels_.shape)
    dir_input = fileinput
    os.makedirs(dir_input, exist_ok=True)
    # np.save(dir_input + 'compounds', compounds)
    # np.save(dir_input + 'adjacencies', adjacencies)
    # np.save(dir_input + 'morgan', morgan)
    # np.save(dir_input + 'proteins_feature', proteins_f)
    # np.save(dir_input + 'interactions', interactions)
    # np.save(dir_input + 'ss_vector', ss_s)
    # np.save(dir_input + 'compound_words', compound_yuyi)
    # np.save(dir_input + 'responses_labels', responses_labels_)


    


if __name__ == "__main__":

    import os
    word_dict = defaultdict(lambda: len(word_dict))
    #zhu("humans/original/data.txt",'humans/input/input_data/')
    #zhu("DB/original/dev.txt",'DB/input/input_dev/')
    # zhu("davis_str_all.txt",'davis_input/')
    zhu("kiba_str_all.txt", 'out/')
    print(len(word_dict))
    print('The preprocess of dataset has finished!')
