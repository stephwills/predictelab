"""
With help from https://colab.research.google.com/drive/1vYq8-MnAckH0oWmmYWbBA2zuqCmQ6Axe?usp=sharing#scrollTo=5jGpFNlm_zkW
"""
import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from src.dataset.pyg_dataset import ElabDataset
from src.model.egnn_clean import EGNN
from torch import nn
from torch_geometric.loader import DataLoader
from src.utils.utils import get_pos_weight, loss_from_avg, score_mol_success_for_batch

# to store loss and activation functions
loss_functions = {'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
                  'BCELoss': nn.BCELoss}  #TODO: change weight calc if want BCELoss?
act_functions = {'SiLU': nn.SiLU}


def run_epoch(model, optim, train_dataloader, eval_dataloader, device, loss_fn='BCEWithLogitsLoss',
              avg_loss_over_mols=False, pos_weight=0):
    """

    :param model:
    :param optim:
    :param train_dataloader:
    :param eval_dataloader:
    :param device:
    :param loss_fn:
    :param avg_loss_over_mols:
    :return:
    """
    # retrieve loss function
    loss_function_init = loss_functions[loss_fn]
    sigmoid = nn.Sigmoid()

    # train the model
    model.train()
    epoch_train_losses, epoch_train_losses_notweighted = [], []
    for i, data in enumerate(train_dataloader):
        data = data.to(device)
        optim.zero_grad()  # delete old gradients

        if avg_loss_over_mols:
            index = data.batch
            y_true = data.y
            out, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)

            if loss_fn == 'BCEWithLogitsLoss':
                loss_function_weighted = loss_function_init(pos_weight=pos_weight, reduction='none')
            if loss_fn == 'BCELoss':
                out = sigmoid(out)
                loss_function_weighted = loss_function_init(weight=pos_weight, reduction='none')

            losses = torch.flatten(loss_function_weighted(out, y_true))
            loss, loss_item = loss_from_avg(losses, index)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optim.step()

            # also record loss without applying weight
            loss_function_notweighted = loss_function_init(reduction='none')
            losses_notweighted = torch.flatten(loss_function_notweighted(out, y_true))
            _, loss_notweighted = loss_from_avg(losses_notweighted, index)
            epoch_train_losses_notweighted.append(loss_notweighted)

        else:
            y_true = data.y
            out, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)

            if loss_fn == 'BCEWithLogitsLoss':
                loss_function_weighted = loss_function_init(pos_weight=pos_weight)
            if loss_fn == 'BCELoss':
                out = sigmoid(out)
                loss_function_weighted = loss_function_init(weight=pos_weight)

            loss = loss_function_weighted(out, y_true)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optim.step()

            # also record loss without applying weight
            loss_function_notweighted = loss_function_init()
            loss_notweighted = loss_function_notweighted(out, y_true)
            epoch_train_losses_notweighted.append(loss_notweighted.item())

    # calculate loss function for validation set
    epoch_val_losses = []
    with torch.no_grad():
        model.eval()  # set the model to eval mode
        for data in eval_dataloader:
            data = data.to(device)

            if avg_loss_over_mols:
                loss_function_notweighted = loss_function_init(reduction='none')
                index = data.batch
                out, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)
                if 'loss_fn' == 'BCELoss':
                    out = sigmoid(out)
                losses = torch.flatten(loss_function_notweighted(out, data.y))
                _, loss = loss_from_avg(losses, index)
                epoch_val_losses.append(loss)

            else:
                loss_function_notweighted = loss_function_init()
                out, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)
                if 'loss_fn' == 'BCELoss':
                    out = sigmoid(out)
                loss = loss_function_notweighted(out, data.y).item()
                epoch_val_losses.append(loss)

    return np.mean(epoch_train_losses), np.mean(epoch_val_losses), np.mean(epoch_train_losses_notweighted)


def test_eval(model, test_loader, device, loss_fn='BCEWithLogitsLoss', avg_loss_over_mols=False):
    """

    :param model:
    :param test_loader:
    :param device:
    :param loss_fn:
    :param avg_loss_over_mols:
    :return:
    """
    # retrieve loss function
    sigmoid = nn.Sigmoid()
    loss_function_init = loss_functions[loss_fn]

    model.eval()  # Set the model to evaluation mode
    test_losses, all_labels, all_predictions, all_probabilities, successes, perc_vectors_found = [], [], [], [], [], []

    with torch.no_grad():  # disable gradient calculation for evaluation

        for data in test_loader:
            data = data.to(device)
            y_true = data.y
            out, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)
            out_sigmoid = sigmoid(out)

            if not avg_loss_over_mols:
                loss_function = loss_function_init()
                if loss_fn == 'BCELoss':
                    loss = loss_function(out_sigmoid, y_true).item()
                if loss_fn == 'BCEWithLogitsLoss':
                    loss = loss_function(out, y_true).item()
                test_losses.append(loss)

            else:
                index = data.batch
                loss_function = loss_function_init(reduction='none')
                if loss_fn == 'BCELoss':
                    losses = torch.flatten(loss_function(out_sigmoid, data.y))
                if loss_fn == 'BCEWithLogitsLoss':
                    losses = torch.flatten(loss_function(out, data.y))
                _, loss = loss_from_avg(losses, index)
                test_losses.append(loss)

            succ, perc_found = score_mol_success_for_batch(data.batch, y_true, out_sigmoid)
            successes.extend(succ)
            perc_vectors_found.extend(perc_found)
            predictions = (out_sigmoid >= 0.5).float()
            all_probabilities.append(out_sigmoid.cpu().detach().numpy())
            all_labels.append(y_true.cpu().detach().numpy())
            all_predictions.append(predictions.cpu().detach().numpy())

    # concatenate all labels and predictions from batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    # Calculate average loss
    avg_loss = np.mean(test_losses)

    # calculate mol successes
    mol_successes = (sum(successes)/len(successes)) * 100

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return avg_loss, accuracy, precision, recall, f1, all_probabilities, mol_successes, perc_vectors_found


def train(n_epochs, patience, lig_codes, mol_files, pdb_files, batch_size, test_size, n_cpus, hidden_nf,
          list_of_vectors=None, random_state=42, lr=1e-4, processed_dir=None, save_processed_files=None, model_dir=None,
          use_wandb=False, project_name='elab_egnn', prot_dist_threshold=8, intra_cutoff=2, inter_cutoff=10,
          verbose=True, avg_loss_over_mols=False, act_fn=nn.SiLU, loss_fn='BCEWithLogitsLoss'):
    """

    :param n_epochs:
    :param patience:
    :param lig_codes:
    :param mol_files:
    :param pdb_files:
    :param batch_size:
    :param test_size:
    :param n_cpus:
    :param hidden_nf:
    :param list_of_vectors:
    :param random_state:
    :param lr:
    :param processed_dir:
    :param save_processed_files:
    :param model_dir:
    :param use_wandb:
    :param project_name:
    :param prot_dist_threshold:
    :param intra_cutoff:
    :param inter_cutoff:
    :param verbose:
    :param avg_loss_over_mols:
    :param act_fn:
    :param loss_fn:
    :return:
    """
    if not model_dir:
        model_dir = '.'

    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=project_name,
            config={"learning_rate": lr,
                    "epochs": n_epochs}
        )

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # load dataset object and run processing (if necessary)
    dataset = ElabDataset(root=None,
                          lig_codes=lig_codes,
                          mol_files=mol_files,
                          pdb_files=pdb_files,
                          list_of_vectors=list_of_vectors,
                          prot_dist_threshold=prot_dist_threshold,
                          intra_cutoff=intra_cutoff,
                          inter_cutoff=inter_cutoff,
                          verbose=verbose,
                          processed_dir=processed_dir,
                          save_processed_files=save_processed_files)
    dataset.load(dataset.processed_file_names[0])

    # split data into train, test and validation sets
    train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)
    train, validation = train_test_split(train, test_size=test_size / 0.95, random_state=random_state)

    # get weight for loss calc
    ys = train.y
    pos_weight = get_pos_weight(ys, is_y=True)
    print('pos weight', pos_weight)

    # create dataloader objects
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=n_cpus,)
    val_dataloader = DataLoader(validation, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)

    # get num of node and edge features
    in_node_nf = dataset[0].x.shape[1]
    out_node_nf = dataset[0].y.shape[1]
    in_edge_nf = dataset[0].edge_attr.shape[1]

    # initialise model
    model = EGNN(in_node_nf, hidden_nf, out_node_nf, in_edge_nf, device, act_fn=act_fn())

    # get optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs_without_improvement = 0

    # run training
    train_losses, val_losses, train_losses_notweighted = [], [], []
    for epoch in range(n_epochs):
        train_loss, val_loss, train_loss_notweighted = run_epoch(model, optim, train_dataloader, val_dataloader,
                                         device=device, loss_fn=loss_fn, avg_loss_over_mols=avg_loss_over_mols, pos_weight=pos_weight)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_losses_notweighted.append(train_loss_notweighted)

        if np.min(val_losses) == val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model"))
            epochs_without_improvement = 0

        elif epochs_without_improvement < patience:
            epochs_without_improvement += 1
        else:
            break

        # EGNNs are quite unstable, this reverts the model to a previous state if an epoch blows up
        if train_loss > 1.5 * np.min(train_losses):
            model.load_state_dict(torch.load(os.path.join(model_dir, "previous_weights"), map_location=torch.device(device)))
            optim.load_state_dict(torch.load(os.path.join(model_dir, "previous_optim"), map_location=torch.device(device)))
        if train_loss == np.min(train_losses):
            torch.save(model.state_dict(), os.path.join(model_dir, "previous_weights"))
            torch.save(optim.state_dict(), os.path.join(model_dir, "previous_optim"))

        print("train {:6.4f} | train (no w) {:6.4f} | val {:6.4f}".format(train_loss, train_loss_notweighted, val_loss))

        if use_wandb:
            run.log({'epoch': epoch,
                     'train_loss': train_loss,
                     'train_loss_noweight': train_loss_notweighted,
                     'val_loss': val_loss})

    avg_loss, accuracy, precision, recall, f1, probabilities, mol_successes, perc_vectors_found = test_eval(model, test_dataloader, device,
                                                                        loss_fn=loss_fn, avg_loss_over_mols=avg_loss_over_mols)

    np.save(os.path.join(model_dir, 'test_set_probabilities.npy'), probabilities)
    with open(os.path.join(model_dir, 'test_set_perc_vectors_found.json'), 'w') as f:
        json.dump(perc_vectors_found, f)

    if use_wandb:
        run.log({'test': {'loss': avg_loss,
                          'accuracy': accuracy,
                          'precision': precision,
                          'recall': recall,
                          'f1': f1,
                          'perc_mol_successes': mol_successes}})

    if use_wandb:
        run.finish()


def main():
    """
    :return:
    """
    import src.utils.config as config
    parser = ArgumentParser()
    parser.add_argument('--data_json', help='dict containing {lig_codes: [], mol_files: [], pdb_files: []}')
    parser.add_argument('--pdb_dir', help='path where the pdb fpaths are located')
    parser.add_argument('--precursor_dir', help='path where the mol fpaths are located')
    parser.add_argument('--processed_dir', help='dir to save processed data')
    parser.add_argument('--model_dir')
    parser.add_argument('--run_name')
    parser.add_argument('--n_epochs', type=int, default=config.N_EPOCHS)
    parser.add_argument('--patience', type=int, default=config.PATIENCE)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--test_size', type=float, default=config.TEST_SIZE)
    parser.add_argument('--n_cpus', type=int, default=config.N_CPUS)
    parser.add_argument('--hidden_nf', type=int, default=config.HIDDEN_NF)
    parser.add_argument('--prot_dist_threshold', type=float, default=config.PROT_DIST_THRESHOLD)
    parser.add_argument('--intra_cutoff', type=float, default=config.INTRA_CUTOFF)
    parser.add_argument('--inter_cutoff', type=float, default=config.INTER_CUTOFF)
    parser.add_argument('--random_state', type=int, default=config.RANDOM_STATE)
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--avg_loss_over_mols', action='store_true')
    parser.add_argument('--loss_function', default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss', 'BCELoss'])
    parser.add_argument('--act_function', default='SiLU', choices=['SiLU'])
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    # save arguments to model dir
    args_dict = vars(args)
    print('ARGUMENTS:')
    print(args_dict)
    with open(os.path.join(args.model_dir, f"{args.run_name}-arguments.json"), "w") as f:
        json.dump(args_dict, f)

    # read in fnames from json file
    with open(args.data_json, "r") as f:
        data = json.load(f)
    lig_codes, mol_files, pdb_files = data['lig_codes'], data['mol_files'], data['pdb_files']
    mol_files = [os.path.join(args.precursor_dir, file) for file in mol_files]
    pdb_files = [os.path.join(args.pdb_dir, file) for file in pdb_files]

    train(n_epochs=args.n_epochs,
          patience=args.patience,
          lig_codes=lig_codes,
          mol_files=mol_files,
          pdb_files=pdb_files,
          batch_size=args.batch_size,
          test_size=args.test_size,
          n_cpus=args.n_cpus,
          hidden_nf=args.hidden_nf,
          list_of_vectors=None,
          random_state=args.random_state,
          lr=args.lr,
          processed_dir=args.processed_dir,
          save_processed_files=True,
          model_dir=args.model_dir,
          use_wandb=args.use_wandb,
          project_name=args.run_name,
          prot_dist_threshold=args.prot_dist_threshold,
          intra_cutoff=args.intra_cutoff,
          inter_cutoff=args.inter_cutoff,
          verbose=args.verbose,
          avg_loss_over_mols=args.avg_loss_over_mols,
          act_fn=act_functions[args.act_function],
          loss_fn=args.loss_function)


if __name__ == "__main__":
    main()
