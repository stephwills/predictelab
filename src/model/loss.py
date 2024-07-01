from torch import nn
import torch
from src.utils.utils import loss_from_avg, rearrange_tensor_for_lig


# to store loss and activation functions
loss_functions = {'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
                  'BCELoss': nn.BCELoss}  #TODO: change weight calc if want BCELoss?


def get_loss(data, loss_fn, pos_weight, loss_type, model, return_out=False):
    """

    :param data:
    :param loss_fn:
    :param pos_weight:
    :param loss_type:
    :param model:
    :param return_out:
    :return:
    """
    sigmoid = nn.Sigmoid()

    if loss_type == 'no_avg':
        reduction = 'mean'
    else:
        reduction = 'none'

    index = data.batch
    y_true = data.y
    out, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)
    out_sigmoid = sigmoid(out)

    loss_function_init = loss_functions[loss_fn]
    if loss_fn == 'BCEWithLogitsLoss':
        loss_function = loss_function_init(pos_weight=pos_weight, reduction=reduction)
    if loss_fn == 'BCELoss':
        out = out_sigmoid
        loss_function = loss_function_init(weight=pos_weight, reduction=reduction)

    if loss_type == 'no_avg':
        loss = loss_function(out, y_true)

    if loss_type == 'avg_over_graph':
        losses = torch.flatten(loss_function(out, y_true))
        loss, _ = loss_from_avg(losses, index)

    if loss_type == 'avg_over_mol':
        losses = torch.flatten(loss_function(out, y_true))
        losses, new_index = rearrange_tensor_for_lig(losses, index, data.lig_mask)
        loss, _ = loss_from_avg(losses, new_index)

    if return_out:
        return loss, out_sigmoid, y_true
    else:
        return loss




