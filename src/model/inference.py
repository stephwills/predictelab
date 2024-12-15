
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from src.dataset.pyg_dataset import ElabDataset
from src.model.loss import *
from src.utils.utils import score_mol_success_for_batch
from src.utils.viz import viz_after_training
from torch_geometric.loader import DataLoader
from src.model.egnn_clean import EGNN


def inference(codes, proc_dir, output_dir, mol_files, pdb_files,
              model_path='/home/swills/Oxford/elaboratability/notebooks/thesis/egnn/model/best_model',
              loss_fn='BCEWithLogitsLoss', loss_type='avg_over_mol', act_fn=nn.SiLU):
    """

    :param codes: ligand codes (unique names)
    :param proc_dir: to save processed data files
    :param output_dir:
    :param mol_files:
    :param pdb_files:
    :param model_path: path for saved model
    :param loss_fn:
    :param loss_type: 
    :param act_fn:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    # mol_files = [os.path.join(malhotra_dir, code, f"{code}_smaller_ligand.sdf") for code in codes]
    # pdb_files = [os.path.join(malhotra_dir, code, f"{code}_larger_receptor.pdb") for code in codes]

    dataset = ElabDataset(root=None,
                              lig_codes=codes,
                              mol_files=mol_files,
                              pdb_files=pdb_files,
                              list_of_vectors=None,
                              prot_dist_threshold=8,
                              intra_cutoff=2,
                              inter_cutoff=10,
                              verbose=True,
                              processed_dir=proc_dir,
                              save_processed_files=True)
    dataset.load(dataset.processed_file_names[0])
    dataloader = DataLoader(dataset, batch_size=8)

    in_node_nf = dataset[0].x.shape[1]
    out_node_nf = dataset[0].y.shape[1]
    in_edge_nf = dataset[0].edge_attr.shape[1]


    model = EGNN(in_node_nf, 30, out_node_nf, in_edge_nf, device, act_fn=act_fn(), n_layers=3)
    model.load_state_dict(torch.load(model_path))
    print('type model', type(model))
    model.eval()

    test_losses, all_labels, all_predictions, all_probabilities, successes, perc_vectors_found = [], [], [], [], [], []

    with torch.no_grad():  # disable gradient calculation for evaluation

        for data in dataloader:
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


    with open(os.path.join(output_dir, 'test_set_perc_vectors_found.json'), 'w') as f:
        json.dump(perc_vectors_found, f)

    # if we want to save visualization
    viz_dir = os.path.join(output_dir, 'viz')
    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    viz_dict = {}
    for i, graph in enumerate(dataset):
        lig_code = graph.lig_code
        lig_mask = graph.lig_mask.cpu().detach().numpy()
        atom_idxs = graph.atom_idxs.cpu().detach().numpy()
        probs = all_probabilities[i]
        y_true = graph.y.cpu().detach().numpy()

        viz_dict[lig_code] = {'lig_mask': lig_mask,
                              'atom_idxs': atom_idxs,
                              'probs': probs,
                              'y_true': y_true}

        viz_after_training(lig_code, codes, mol_files, pdb_files, probs, lig_mask, atom_idxs, viz_dir, loss_type)

    # print(viz_dict)
    np.savez(os.path.join(output_dir, 'test_set_visualization.npz'), **viz_dict)
