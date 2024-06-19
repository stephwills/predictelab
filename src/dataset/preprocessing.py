import numpy as np
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial.distance import cdist

PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                   'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                   'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']


def get_protein_coords(pdb):
    """

    :param pdb:
    :return:
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb)
    prot_atoms = [x for x in structure.get_atoms()]
    prot_coords = np.array([a.coord for a in prot_atoms], dtype="d")
    prot_elems = np.array([a.element for a in prot_atoms])
    return prot_coords, prot_elems


def get_ligand_coords(file, returnVectors=False):
    """

    :param sdf:
    :param isVector:
    :param vectorAtoms:
    :return:
    """
    ext = file.split('.')[-1]
    if ext == 'sdf':
        mol = Chem.SDMolSupplier(file)[0]
    if ext == 'mol':
        mol = Chem.MolFromMolFile(file)

    mol = Chem.AddHs(mol, addCoords=True)

    if returnVectors:
        vectors = []
        if mol.HasProp('vectors'):
            vectors = mol.GetProp('vectors').split(',')
            vectors = list(map(int, vectors))
        else:
            # print('No vectors for', file)
            vectors = []

    conf = mol.GetConformer()

    lig_coords, lig_elems, lig_idxs = [], [], []

    for idx in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1:
            lig_coords.append(np.array(conf.GetAtomPosition(idx)))
            lig_elems.append(mol.GetAtomWithIdx(idx).GetSymbol())
            lig_idxs.append(idx)

    lig_coords = np.array(lig_coords)
    lig_elems = np.array(lig_elems)

    if returnVectors:
        return lig_idxs, lig_elems, lig_coords, vectors

    return lig_idxs, lig_elems, lig_coords


def one_hot_encoding(item, permitted_list):
    """

    :param elem:
    :param atom_types:
    :return:
    """
    if item not in permitted_list:
        item = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: item == s, permitted_list))]
    return binary_encoding


def get_atom_feature_vector(elem, isProt):
    """

    :param elem:
    :param isProt:
    :return:
    """
    one_hot = one_hot_encoding(elem, PERMITTED_ATOMS)
    feature_vector = one_hot + [int(isProt)]
    return np.array(feature_vector)


def get_ligand_bond_features(bond, use_stereochemistry=True):
    """
    NOT USED; https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/

    :param bond:
    :param use_stereochemistry:
    :return:
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def get_edge_feature_vector(i_is_prot, j_is_prot):
    """

    :param i_is_prot:
    :param j_is_prot:
    :return:
    """
    # feature vector: both atoms in protein, both atoms in ligand, one atom in ligand and one in protein
    if i_is_prot and j_is_prot:
        return np.array([1, 0, 0])
    if not i_is_prot and not j_is_prot:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


def get_edges(ligand_coords, protein_coords, intra_cutoff=2.0, inter_cutoff=10.0):
    """

    :param ligand_coords:
    :param protein_coords:
    :param intra_cutoff:
    :param inter_cutoff:
    :return:
    """
    all_coords = np.vstack((ligand_coords, protein_coords))
    is_prot = len(ligand_coords) * [True] + len(protein_coords) * [False]

    atom_dists = cdist(all_coords, all_coords)

    edge_index = []
    edge_feature_vectors = []

    inter_dists = np.where(atom_dists < inter_cutoff)

    for i, j in zip(inter_dists[0], inter_dists[1]):
        if i != j:
            if is_prot[i] != is_prot[j]:
                edge_index.append((i, j))
                edge_feature_vectors.append(get_edge_feature_vector(is_prot[i], is_prot[j]))
    intra_dists = np.where(atom_dists < intra_cutoff)
    for i, j in zip(intra_dists[0], intra_dists[0]):
        if i != j:
            if is_prot[i] == is_prot[j]:
                edge_index.append((i, j))
                edge_feature_vectors.append(get_edge_feature_vector(is_prot[i], is_prot[j]))

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_feature_vectors = torch.tensor(list(edge_feature_vectors))
    return edge_index, edge_feature_vectors


def convert_to_graph(mol_file, pdb, prot_dist_threshold=8, intra_cutoff=2.0, inter_cutoff=10.0, vectors=None,
                     vectors_are_molprop=False):
    """
    Given a file for the ligand and a separate apo file for the protein, create input graph to GNN. Considers only
    'pocket atoms' within a specified distance of a ligand atom. Edges are decided according to distance between atoms
    (and features denote what molecule they are inbetween).


    :param mol_file:
    :param pdb:
    :param prot_dist_threshold:
    :param intra_cutoff:
    :param inter_cutoff:
    :return:
    """
    if vectors_are_molprop:
        lig_idxs, lig_elems, lig_coords, vector_idxs = get_ligand_coords(mol_file, returnVectors=True)
    else:
        lig_idxs, lig_elems, lig_coords = get_ligand_coords(mol_file, returnVectors=False)
        vector_idxs = vectors

    lig_vectors = [get_atom_feature_vector(elem, False) for elem in lig_elems]
    prot_coords, prot_elems = get_protein_coords(pdb)
    dists = cdist(lig_coords, prot_coords)
    dist_thresh = np.where(dists < prot_dist_threshold)

    select_prot_idxs = set()
    for prot_idx in dist_thresh[1]:
        select_prot_idxs.add(prot_idx)
    select_prot_idxs = list(select_prot_idxs)

    prot_coords = prot_coords[select_prot_idxs]
    prot_elems = prot_elems[select_prot_idxs]
    prot_vectors = [get_atom_feature_vector(elem, True) for elem in prot_elems]

    feat_vectors = lig_vectors + prot_vectors

    n_nodes = len(feat_vectors)
    n_node_features = len(feat_vectors[0])

    h = np.zeros((n_nodes, n_node_features))
    for i, feat_vector in enumerate(feat_vectors):
        h[i, :] = feat_vector
    h = torch.tensor(h, dtype=torch.float)

    y = [int(idx in vector_idxs) for idx in lig_idxs] + ([0] * len(prot_vectors))
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    x = np.vstack((lig_coords, prot_coords))
    x = torch.tensor(x, dtype=torch.float)

    edge_index, edge_attr = get_edges(lig_coords, prot_coords, intra_cutoff, inter_cutoff)
    return h, y, x, edge_index, edge_attr
