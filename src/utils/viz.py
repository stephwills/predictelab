"""Code for visualizing results"""
import os

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from rdkit import Chem


def process_for_viz(pdb_file, lig_sdf, probabilities, lig_mask, atom_idxs, target_file):
    """

    :param pdb_file:
    :param lig_sdf:
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
    lig = Chem.AddHs(Chem.SDMolSupplier(lig_sdf)[0], addCoords=True)
    tmp_file = os.path.join('/tmp', os.path.basename(lig_sdf).replace('.sdf', '.pdb'))
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


def read_data_from_npz(file, lig_code):
    """

    :param file:
    :param lig_code:
    :return:
    """
    data = np.load(file, allow_pickle=True)
    file = data.files[0]
    data_dict = data[file].item()
    lig_mask = data_dict[lig_code]['lig_mask']
    atom_idxs = data_dict[lig_code]['atom_idxs']
    probs = data_dict[lig_code]['probs']
    return lig_mask, atom_idxs, probs
