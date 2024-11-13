import os

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_fnames_from_dir(dir):
    """

    :param dir:
    :return:
    """
    lig_codes = [code for code in os.listdir(dir)]
    mol_files = [os.path.join(dir, code, f"{code}.sdf") for code in lig_codes]
    receptor_files = [os.path.join(dir, code, f"{code}_receptor.pdb") for code in lig_codes]
    return lig_codes, mol_files, receptor_files


def get_pos_weight_from_train(train, lig_only=False):
    """

    :param train:
    :return:
    """
    if not lig_only:
        num_0s = sum([(data.y == 0).sum() for data in train]).item()
        num_1s = sum([(data.y).sum() for data in train]).item()
        return torch.tensor(num_0s / num_1s, dtype=torch.float64, device=device)
    else:
        num_0s = sum([(mask_split(data.y, data.lig_mask)[1] == 0).sum() for data in train]).item()
        num_1s = sum([(mask_split(data.y, data.lig_mask)[1]).sum() for data in train]).item()
        return torch.tensor(num_0s / num_1s, dtype=torch.float64, device=device)


def get_pos_weight(data, is_y=False):
    """
    Not used

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


def rearrange_tensor_for_lig(losses, index, mol_index):
    """

    :param losses:
    :param index:
    :param mol_index:
    :return:
    """
    split_losses = mask_split(losses, index)
    split_idxs = mask_split(index, index)
    split_mol_idxs = mask_split(mol_index, index)

    lig_losses_all = torch.tensor([], device=device)
    index_all = []

    for split_loss, split_idx, split_mol_idx in zip(split_losses, split_idxs, split_mol_idxs):
        lig_losses = mask_split(split_loss, split_mol_idx)[1]
        lig_idx = mask_split(split_idx, split_mol_idx)[1]
        lig_losses_all = torch.cat((lig_losses_all, lig_losses))
        index_all.extend(lig_idx)

    return lig_losses_all, torch.tensor(index_all, device=device)


def mask_avg(src, index):
    """
    Get avg according to index

    :param src:
    :param index:
    :return:
    """
    uniq_elements, counts = torch.unique(index, return_counts=True)
    n_indices = uniq_elements.numel()
    added = torch.zeros(n_indices, dtype=src.dtype, device=device)
    added = torch.scatter_add(added, dim=0, index=index, src=src)
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


def score_mol_success(y_true, probabilities, quantile=0.9):
    """
    Get whether the true vectors are in the top X% of predictions from the model.
    Returns true if all are in the top X%, or False if not (plus the perc that were correct).

    :return:
    """
    threshold = torch.quantile(probabilities, quantile)
    idxs = torch.nonzero(probabilities >= threshold).flatten()
    y_idxs = torch.nonzero(y_true.flatten() == 1.0).flatten()
    num_correct = torch.isin(y_idxs, idxs).sum().item()
    num_poss = y_idxs.numel()
    if num_correct == num_poss:
        return True, 1.0
    else:
        return False, num_correct/num_poss


def score_mol_success_for_batch(data, y_true, probabilities, quantile=0.75, type_calc='avg_over_mol'):
    """
    score_mol_success but over a batch

    :param batch:
    :param y_true:
    :param probabilities:
    :param quantile:
    :return:7
    """
    batch = data.batch
    mol_index = data.lig_mask
    successes, perc_found = [], []

    if type_calc == 'avg_over_mol':
        probabilities, new_batch = rearrange_tensor_for_lig(probabilities, batch, mol_index)
        y_true, _ = rearrange_tensor_for_lig(y_true, batch, mol_index)
        batch = new_batch

    probs_split = mask_split(probabilities, batch)
    y_true_split = mask_split(y_true, batch)
    for probs, y_true in zip(probs_split, y_true_split):
        res, perc = score_mol_success(y_true, probs, quantile)
        successes.append(res)
        perc_found.append(perc)
    return successes, perc_found, probs_split


def get_metrics_from_predictions(all_labels, all_predictions, successes):
    # concatenate all labels and predictions from batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    # calculate mol successes
    mol_successes = (sum(successes) / len(successes)) * 100

    return accuracy, precision, recall, f1, mol_successes
