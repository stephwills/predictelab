"""Code for visualizing results"""
import os

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from rdkit import Chem


def process_for_viz(pdb_file, mol_file, probabilities, lig_mask, atom_idxs, target_file):
    """

    :param pdb_file:
    :param mol_file:
    :param probabilities:
    :param lig_mask:
    :param atom_idxs:
    :param target_file:
    :return:
    """
    parser = PDBParser(QUIET=True)

    # process protein pdb
    structure1 = parser.get_structure('prot', pdb_file)
    prot_atoms = [x for x in structure1.get_atoms()]

    prot_probs_ordered = []
    prot_probabilities = [prob[0] for prob, mask in zip(probabilities, lig_mask) if mask != 1]
    prot_idxs = [idx for idx, mask in zip(atom_idxs, lig_mask) if mask != 1]

    for idx in range(len(prot_atoms)):
        if idx in prot_idxs:
            prob = prot_probabilities[prot_idxs.index(idx)]
            prot_probs_ordered.append(prob)
        else:
            prot_probs_ordered.append(0)

    for atom, prob in zip(prot_atoms, prot_probs_ordered):
        atom.set_bfactor(prob)

    # process ligand
    # convert ligand sdf to pdb
    if '.sdf' in os.path.basename(mol_file):
        ext = '.sdf'
        lig = Chem.AddHs(Chem.SDMolSupplier(mol_file)[0], addCoords=True)
    if '.mol' in os.path.basename(mol_file):
        ext = '.mol'
        lig = Chem.AddHs(Chem.MolFromMolFile(mol_file)[0], addCoords=True)

    tmp_file = os.path.join('/tmp', os.path.basename(mol_file).replace(ext, '.pdb'))
    Chem.MolToPDBFile(lig, tmp_file)

    structure2 = parser.get_structure('lig', tmp_file)
    lig_atoms = [x for x in structure2.get_atoms()]

    lig_probs_ordered = []
    lig_probabilities = [prob[0] for prob, mask in zip(probabilities, lig_mask) if mask == 1]
    lig_idxs = [idx for idx, mask in zip(atom_idxs, lig_mask) if mask == 1]

    for idx in range(len(lig_atoms)):
        if idx in lig_idxs:
            prob = lig_probabilities[lig_idxs.index(idx)]
            lig_probs_ordered.append(prob)
        else:
            lig_probs_ordered.append(0)

    for atom, prob in zip(lig_atoms, lig_probs_ordered):
        atom.bfactor = prob

    io = PDBIO()
    ligand_chain = list(structure2.get_chains())[0]
    ligand_chain.id = 'L'
    ligand_chain.detach_parent()
    structure1[0].add(ligand_chain)
    io.set_structure(structure1)
    io.save(target_file)
    os.remove(tmp_file)


def data_from_dict(data_dict, lig_code):
    """
    Get rel data for creating viz

    :param data_dict:
    :param lig_code:
    :return:
    """
    lig_mask = data_dict[lig_code]['lig_mask']
    atom_idxs = data_dict[lig_code]['atom_idxs']
    probs = data_dict[lig_code]['probs']
    return lig_mask, atom_idxs, probs


def data_dict_from_npz(file):
    """
    Read npz file

    :param file:
    :return:
    """
    data = np.load(file, allow_pickle=True)
    file = data.files[0]
    data_dict = data[file].item()
    return data_dict


def main():
    """

    :return:
    """
    import json
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--precursor_dir')
    parser.add_argument('--pdb_dir')
    parser.add_argument('--data_json', help='contains the names of all the mol and pdb files')
    parser.add_argument('--npz_file', help='the test_set_visualization.npz file saved by train.py')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(args.data_json, "r") as f:
        data = json.load(f)
    lig_codes, mol_files, pdb_files = data['lig_codes'], data['mol_files'], data['pdb_files']

    mol_files = [os.path.join(args.precursor_dir, file) for file in mol_files]
    pdb_files = [os.path.join(args.pdb_dir, file) for file in pdb_files]

    data_dict = data_dict_from_npz(args.npz_file)
    test_lig_codes = list(data_dict.keys())
    test_lig_codes.sort()

    for test_lig_code in test_lig_codes:
        lig_mask, atom_idxs, probs = data_from_dict(data_dict, test_lig_code)
        lig_idx = lig_codes.index(test_lig_code)
        mol_file, pdb_file = mol_files[lig_idx], pdb_files[lig_idx]
        target_file = os.path.join(args.output_dir, f"{test_lig_code}_viz.pdb")
        process_for_viz(pdb_file, mol_file, probs, lig_mask, atom_idxs, target_file)


if __name__ == "__main__":
    main()