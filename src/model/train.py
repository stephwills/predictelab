"""
With help from https://colab.research.google.com/drive/1vYq8-MnAckH0oWmmYWbBA2zuqCmQ6Axe?usp=sharing#scrollTo=5jGpFNlm_zkW
"""
import json
import os
from argparse import ArgumentParser

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from src.dataset.pyg_dataset import ElabDataset
from src.model.egnn_clean import EGNN
from src.model.loss import *
from src.utils.utils import (get_metrics_from_predictions,
                             get_pos_weight_from_train,
                             score_mol_success_for_batch)
from src.utils.viz import viz_after_training
from torch_geometric.loader import DataLoader

act_functions = {'SiLU': nn.SiLU}


def run_epoch(epoch, model, optim, train_dataloader, eval_dataloader, device, loss_fn='BCEWithLogitsLoss',
              pos_weight=None, loss_type='no_avg', use_lr_scheduler=False, scheduler=None):
    """

    :param model:
    :param optim:
    :param train_dataloader:
    :param eval_dataloader:
    :param device:
    :param loss_fn:
    :param pos_weight:
    :param loss_type:
    :return:
    """
    # train the model
    model.train()
    epoch_train_losses, epoch_train_losses_notweighted = [], []
    train_labels, train_predictions, train_successes = [], [], []

    for i, data in enumerate(train_dataloader):
        data = data.to(device)
        optim.zero_grad()  # delete old gradients
        loss, out_sigmoid, y_true = get_loss(data, loss_fn, pos_weight, loss_type, model, return_out=True)
        loss_not_weighted = get_loss(data, loss_fn, None, loss_type, model)
        epoch_train_losses.append(loss.item())
        epoch_train_losses_notweighted.append(loss_not_weighted.item())
        loss.backward()
        optim.step()

        # calculate metrics
        succ, perc_found, probabilities_split = score_mol_success_for_batch(data, y_true, out_sigmoid,
                                                                            type_calc=loss_type)
        train_successes.extend(succ)
        predictions = (out_sigmoid >= 0.5).float()
        train_labels.append(y_true.cpu().detach().numpy())
        train_predictions.append(predictions.cpu().detach().numpy())

    if use_lr_scheduler:
        before_lr = optim.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optim.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

    # calculate loss function for validation set
    epoch_val_losses = []
    val_labels, val_predictions, val_successes = [], [], []
    with torch.no_grad():
        model.eval()  # set the model to eval mode
        for data in eval_dataloader:
            data = data.to(device)
            val_loss, val_out_sigmoid, val_y_true = get_loss(data, loss_fn, None, loss_type, model, return_out=True)
            epoch_val_losses.append(val_loss.item())

            # calculate metrics
            v_succ, v_perc_found, v_probabilities_split = score_mol_success_for_batch(data, val_y_true, val_out_sigmoid,
                                                                                type_calc=loss_type)
            val_successes.extend(v_succ)
            v_predictions = (val_out_sigmoid >= 0.5).float()
            val_labels.append(val_y_true.cpu().detach().numpy())
            val_predictions.append(v_predictions.cpu().detach().numpy())

    # concatenate all labels and predictions from batches
    train_labels = np.concatenate(train_labels)
    train_predictions = np.concatenate(train_predictions)
    train_accuracy, train_precision, train_recall, train_f1, train_mol_successes = get_metrics_from_predictions(train_labels, train_predictions, train_successes)

    val_labels = np.concatenate(val_labels)
    val_predictions = np.concatenate(val_predictions)
    val_accuracy, val_precision, val_recall, val_f1, val_mol_successes = get_metrics_from_predictions(val_labels, val_predictions, val_successes)


    return (np.mean(epoch_train_losses), np.mean(epoch_val_losses), np.mean(epoch_train_losses_notweighted), \
            train_accuracy, train_precision, train_recall, train_f1, train_mol_successes, \
            val_accuracy, val_precision, val_recall, val_f1, val_mol_successes)


def test_eval(model, test_loader, device, loss_fn='BCEWithLogitsLoss', loss_type='no_avg'):
    """

    :param model:
    :param test_loader:
    :param device:
    :param loss_fn:
    :param loss_type:
    :return:
    """
    model.eval()  # Set the model to evaluation mode
    test_losses, all_labels, all_predictions, all_probabilities, successes, perc_vectors_found = [], [], [], [], [], []

    with torch.no_grad():  # disable gradient calculation for evaluation

        for data in test_loader:
            data = data.to(device)
            test_loss, out_sigmoid, y_true = get_loss(data, loss_fn, None, loss_type, model, return_out=True)
            test_losses.append(test_loss.item())

            succ, perc_found, probabilities_split = score_mol_success_for_batch(data, y_true, out_sigmoid, type_calc=loss_type)
            successes.extend(succ)
            perc_vectors_found.extend(perc_found)
            predictions = (out_sigmoid >= 0.5).float()
            for probs in probabilities_split:
                probs = probs.flatten()
                all_probabilities.append(probs.cpu().detach().numpy())

            all_labels.append(y_true.cpu().detach().numpy())
            all_predictions.append(predictions.cpu().detach().numpy())

    # concatenate all labels and predictions from batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

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
    print(f"Mol successes: {mol_successes:.4f}")

    return avg_loss, accuracy, precision, recall, f1, all_probabilities, mol_successes, perc_vectors_found


def train(n_epochs, patience, lig_codes, mol_files, pdb_files, batch_size, test_size, n_cpus, hidden_nf,
          list_of_vectors=None, random_state=42, lr=1e-4, processed_dir=None, save_processed_files=None, model_dir=None,
          use_wandb=False, project_name='elab_egnn', prot_dist_threshold=8, intra_cutoff=2, inter_cutoff=10,
          verbose=True, act_fn=nn.SiLU, loss_fn='BCEWithLogitsLoss', loss_type='no_avg', n_layers=4,
          use_lr_scheduler=False, lr_scheduler_type='Linear'):
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

    # create dataloader objects
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=n_cpus,)
    val_dataloader = DataLoader(validation, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)

    # get weight for loss calc
    if loss_type == 'avg_over_mol':
        pos_weight = get_pos_weight_from_train(train, lig_only=True)
    else:
        pos_weight = get_pos_weight_from_train(train)
    print('pos weight', pos_weight)

    # get num of node and edge features
    in_node_nf = dataset[0].x.shape[1]
    out_node_nf = dataset[0].y.shape[1]
    in_edge_nf = dataset[0].edge_attr.shape[1]

    # initialise model
    model = EGNN(in_node_nf, hidden_nf, out_node_nf, in_edge_nf, device, act_fn=act_fn(), n_layers=n_layers)

    # get optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs_without_improvement = 0

    if use_lr_scheduler:
        if lr_scheduler_type == 'Linear':
            scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.3, total_iters=10)
        if lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, threshold=0.0001)

    # run training
    train_losses, val_losses, train_losses_notweighted = [], [], []
    for epoch in range(n_epochs):
        epoch_data = run_epoch(epoch, model, optim, train_dataloader, val_dataloader,
                                         device=device, loss_fn=loss_fn, pos_weight=pos_weight,
                                         loss_type=loss_type, use_lr_scheduler=use_lr_scheduler,
                                         scheduler=scheduler)

        train_loss, val_loss, train_loss_notweighted = epoch_data[0], epoch_data[1], epoch_data[2]
        train_accuracy, train_precision, train_recall, train_f1, train_mol_successes = epoch_data[3], epoch_data[4], epoch_data[5], epoch_data[6], epoch_data[7]
        val_accuracy, val_precision, val_recall, val_f1, val_mol_successes = epoch_data[8], epoch_data[9], epoch_data[10], epoch_data[11], epoch_data[12]

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
                     'val_loss': val_loss,
                     'train_accuracy': train_accuracy,
                     'train_precision': train_precision,
                     'train_recall': train_recall,
                     'train_f1': train_f1,
                     'train_mol_successes': train_mol_successes,
                     'val_accuracy': val_accuracy,
                     'val_precision': val_precision,
                     'val_recall': val_recall,
                     'val_f1': val_f1,
                     'val_mol_successes': val_mol_successes})

    avg_loss, accuracy, precision, recall, f1, probabilities, mol_successes, perc_vectors_found = test_eval(model, test_dataloader, device,
                                                                        loss_fn=loss_fn, loss_type=loss_type)

    with open(os.path.join(model_dir, 'test_set_perc_vectors_found.json'), 'w') as f:
        json.dump(perc_vectors_found, f)

    if use_wandb:
        print('logging test')
        run.log({'test': {'loss': avg_loss,
                          'accuracy': accuracy,
                          'precision': precision,
                          'recall': recall,
                          'f1': f1,
                          'perc_mol_successes': mol_successes}})

    # if we want to save visualization
    viz_dir = os.path.join(model_dir, 'viz')
    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    viz_dict = {}
    for i, graph in enumerate(test):
        lig_code = graph.lig_code
        lig_mask = graph.lig_mask.cpu().detach().numpy()
        atom_idxs = graph.atom_idxs.cpu().detach().numpy()
        probs = probabilities[i]
        y_true = graph.y.cpu().detach().numpy()

        viz_dict[lig_code] = {'lig_mask': lig_mask,
                              'atom_idxs': atom_idxs,
                              'probs': probs,
                              'y_true': y_true}

        viz_after_training(lig_code, lig_codes, mol_files, pdb_files, probs, lig_mask, atom_idxs, viz_dir, loss_type)

    # print(viz_dict)
    np.savez(os.path.join(model_dir, 'test_set_visualization.npz'), **viz_dict)

    if use_wandb:
        print('finish')
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
    parser.add_argument('--loss_type', default='no_avg', choices=['no_avg', 'avg_over_graph', 'avg_over_mol'])
    parser.add_argument('--loss_function', default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss', 'BCELoss'])
    parser.add_argument('--act_function', default='SiLU', choices=['SiLU'])
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_lr_scheduler', action='store_true')
    parser.add_argument('--lr_scheduler_type', default='ReduceLROnPlateau', choices=['ReduceLROnPlateau', 'Linear'])
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
          loss_type=args.loss_type,
          act_fn=act_functions[args.act_function],
          loss_fn=args.loss_function,
          n_layers=args.n_layers,
          use_lr_scheduler=args.use_lr_scheduler,
          lr_scheduler_type=args.lr_scheduler_type)


if __name__ == "__main__":
    main()
