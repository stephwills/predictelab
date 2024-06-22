import os
import torch


def generate_fnames_from_dir(dir):
    lig_codes = [code for code in os.listdir(dir)]
    mol_files = [os.path.join(dir, code, f"{code}.sdf") for code in lig_codes]
    receptor_files = [os.path.join(dir, code, f"{code}_receptor.pdb") for code in lig_codes]
    return lig_codes, mol_files, receptor_files


def get_pos_weight(data, is_y=False):
    if not is_y:
        y = data.y
    else:
        y = data
    pos_weight = (y==0.).sum()/y.sum()
    return pos_weight


def mask_split(tensor, indices):
    unique = torch.unique(indices)
    return [tensor[indices == i] for i in unique]


def mask_avg(src, index):
    uniq_elements, counts = torch.unique(index, return_counts=True)
    n_indices = uniq_elements.numel()
    added = torch.zeros(n_indices, dtype=src.dtype)
    added = torch.scatter(added, dim=0, index=index, src=src)
    return added / counts
