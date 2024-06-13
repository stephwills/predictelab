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
from src.dataset.dataset import ElabDataset
from src.model.egnn_clean import EGNN
from torch import nn
from torch.nn import BCELoss
from torch_geometric.loader import DataLoader


def run_epoch(model, optim, train_dataloader, eval_dataloader, loss_fn=BCELoss()):
    """

    :param model:
    :param optim:
    :param train_dataloader:
    :param eval_dataloader:
    :param loss_fn:
    :return:
    """
    epoch_train_losses = []
    model.train()

    for i, data in enumerate(train_dataloader):  # for each batch
        optim.zero_grad()  # delete old gradients

        y_true = data.y
        y_pred, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)

        loss = loss_fn(y_pred, y_true)
        epoch_train_losses.append(loss.detach().numpy())
        loss.backward()
        optim.step()

    with torch.no_grad():  # Calculate loss function for validation set
        model.eval()  # Set the model to eval mode
        epoch_val_loss = np.mean([loss_fn(model(data.x, data.pos, data.edge_index, data.edge_attr)[0], data.y) for data in eval_dataloader])

    return np.mean(epoch_train_losses), epoch_val_loss


def test_eval(model, test_loader, loss_fn=BCELoss()):
    """

    :param model:
    :param test_loader:
    :param loss_fn:
    :return:
    """
    model.eval()  # Set the model to evaluation mode
    test_losses = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data in test_loader:
            y_true = data.y
            y_pred, _ = model(data.x, data.pos, data.edge_index, data.edge_attr)
            loss = loss_fn(y_pred, y_true)
            test_losses.append(loss.detach().numpy())

            # Convert outputs to binary predictions
            predictions = (y_pred >= 0.5).float()

            all_labels.append(y_true)
            all_predictions.append(np.array(predictions))

    # Concatenate all labels and predictions from batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    # Calculate average loss
    avg_loss = np.mean(test_losses)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return avg_loss, accuracy, precision, recall, f1


def train(n_epochs, patience, lig_codes, mol_files, pdb_files, batch_size, test_size, n_cpus, hidden_nf, list_of_vectors=None, random_state=42, lr=1e-4,
         processed_dir=None, save_processed_files=None, model_dir=None, use_wandb=False, project_name='elab_egnn',
         prot_dist_threshold=8, intra_cutoff=2, inter_cutoff=10, mol_file_suffix='.mol', pdb_file_suffix='_receptor.pdb',
         verbose=True, loss_fn=BCELoss()):
    """

    :param n_epochs:
    :param patience:
    :param data_dir:
    :param batch_size:
    :param test_size:
    :param n_cpus:
    :param hidden_nf:
    :param lig_codes:
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
    :param mol_file_suffix:
    :param pdb_file_suffix:
    :param verbose:
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

    # split data
    dataset = ElabDataset(lig_codes=lig_codes,
                          mol_files=mol_files,
                          pdb_files=pdb_files,
                          list_of_vectors=list_of_vectors,
                          prot_dist_threshold=prot_dist_threshold,
                          intra_cutoff=intra_cutoff,
                          inter_cutoff=inter_cutoff,
                          mol_file_suffix=mol_file_suffix,
                          pdb_file_suffix=pdb_file_suffix,
                          verbose=verbose,
                          processed_dir=processed_dir,
                          save_processed_files=save_processed_files)
    train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)
    train, validation = train_test_split(train, test_size=test_size / 0.95, random_state=random_state)

    # create dataloader
    train_dataloader = DataLoader(train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_cpus)
    val_dataloader = DataLoader(validation, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)

    train_losses, val_losses = [], []

    in_node_nf = 44
    out_node_nf = 1
    in_edge_nf = 3
    model = EGNN(in_node_nf, hidden_nf, out_node_nf, in_edge_nf, device, act_fn=nn.Sigmoid())

    # get optim
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        train_loss, val_loss = run_epoch(model, optim, train_dataloader, val_dataloader, loss_fn=loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if np.min(val_losses) == val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model"))
            epochs_without_improvement = 0

        elif epochs_without_improvement < patience:
            epochs_without_improvement += 1
        else:
            break

        if train_loss > 1.5 * np.min(
                train_losses):  # EGNNs are quite unstable, this reverts the model to a previous state if an epoch blows up
            model.load_state_dict(torch.load(os.path.join(model_dir, "previous_weights"), map_location=torch.device(device)))
            optim.load_state_dict(torch.load(os.path.join(model_dir, "previous_optim"), map_location=torch.device(device)))
        if train_loss == np.min(train_losses):
            torch.save(model.state_dict(), os.path.join(model_dir, "previous_weights"))
            torch.save(optim.state_dict(), os.path.join(model_dir, "previous_optim"))

        print("{:6.2f} | {:6.2f}".format(train_loss, val_loss))

        if use_wandb:
            run.log({'epoch': epoch,
                     'train_loss': train_loss,
                     'val_loss': val_loss})

    avg_loss, accuracy, precision, recall, f1 = test_eval(model, test_dataloader, loss_fn=loss_fn)

    if use_wandb:
        run.log({'test': {'loss': avg_loss,
                          'accuracy': accuracy,
                          'precision': precision,
                           'recall': recall,
                           'f1': f1}})

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
    parser.add_argument('--mol_file_suffix', default=config.MOL_FILE_SUFFIX)
    parser.add_argument('--pdb_file_suffix', default=config.PDB_FILE_SUFFIX)
    parser.add_argument('--random_state', type=int, default=config.RANDOM_STATE)
    parser.add_argument('--lr', type=float, default=config.LR)
    args = parser.parse_args()

    # save arguments to model dir
    args_dict = vars(args)
    print('ARGUMENTS:')
    print(args_dict)
    with open(os.path.join(args.model_dir, f"{args.run_name}-arguments.json"), "w") as f:
        json.dump(args_dict, f)

    # read in a dictionary containing the ligand codes and fnames of data to process
    # TODO: consider data may be processed already
    with open(args.data_json, "r") as f:
        data = json.load(f)
    lig_codes, mol_files, pdb_files = data['lig_codes'], data['mol_files'], data['pdb_files']
    mol_files = [os.path.join(args.precursor_dir, file) for file in mol_files]
    pdb_files = [os.path.join(args.pdb_dir, file) for file in pdb_files]

    train(args.n_epochs, args.patience, lig_codes, mol_files, pdb_files, args.batch_size, args.test_size, args.n_cpus,
          args.hidden_nf, list_of_vectors=None, random_state=args.random_state, lr=args.lr, processed_dir=args.processed_dir,
          save_processed_files=True, model_dir=args.model_dir, use_wandb=True, project_name=args.run_name, prot_dist_threshold=args.prot_dist_threshold,
          intra_cutoff=args.intra_cutoff, inter_cutoff=args.inter_cutoff, mol_file_suffix=args.mol_file_suffix, pdb_file_suffix=args.pdb_file_suffix,
          verbose=True)


if __name__ == "__main__":
    main()
