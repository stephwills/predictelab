import os
import torch


def generate_fnames_from_dir(dir):
    """

    :param dir:
    :return:
    """
    lig_codes = [code for code in os.listdir(dir)]
    mol_files = [os.path.join(dir, code, f"{code}.sdf") for code in lig_codes]
    receptor_files = [os.path.join(dir, code, f"{code}_receptor.pdb") for code in lig_codes]
    return lig_codes, mol_files, receptor_files


def get_pos_weight(data, is_y=False):
    """

    :param data:
    :param is_y:
    :return:
    """
    if not is_y:
        y = data.y
    else:
        y = data
    pos_weight = (y==0.).sum()/y.sum()
    return pos_weight


def mask_split(tensor, indices):
    """
    Not used: split tensor according to index

    :param tensor:
    :param indices:
    :return:
    """
    unique = torch.unique(indices)
    return [tensor[indices == i] for i in unique]


def mask_avg(src, index):
    """
    Get avg according to index

    :param src:
    :param index:
    :return:
    """
    uniq_elements, counts = torch.unique(index, return_counts=True)
    n_indices = uniq_elements.numel()
    added = torch.zeros(n_indices, dtype=src.dtype)
    added = torch.scatter(added, dim=0, index=index, src=src)
    return added / counts


def loss_from_avg(losses, index):
    """
    Get the avg loss when calculated over graphs rather than all nodes in batch

    :param losses:
    :param index:
    :return:
    """
    avg_losses = mask_avg(losses, index)
    loss = torch.mean(avg_losses)
    loss_item = loss.item()
    return loss, loss_item
