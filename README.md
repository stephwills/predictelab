# EGNN for elaboratability prediction

Contains code for training an EGNN to predict the elaboratability of a compound.
The model is trained using synthetic data generated using AiZynthFinder.
The model is trained using PyTorch and PyTorchGeometric.

## Install

To install the necessary packages:
```
conda create -n predictelab python=3.9 -y
conda activate predictelab
conda install -c conda-forge rdkit -y
conda install -c conda-forge numpy -y
conda install -c conda-forge sklearn -y
conda install -c conda-forge scipy -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pyg -c pyg -y
pip install joblib tqdm wandb biopython
```

## Usage

### Dataset

The pytorch geometric dataset is created by `src/dataset/pyg_dataset.py`. A possible change to make is how bonds are encoded so the model can learn more chemistry (currently just based on distance thresholds and edge features represent whether edge is between protein-protein, protein-ligand, ligand-ligand).

### Train

```
usage: train.py [-h] [--data_json DATA_JSON] [--pdb_dir PDB_DIR] [--precursor_dir PRECURSOR_DIR] [--processed_dir PROCESSED_DIR] [--model_dir MODEL_DIR] [--run_name RUN_NAME] [--n_epochs N_EPOCHS] [--patience PATIENCE]
                [--batch_size BATCH_SIZE] [--test_size TEST_SIZE] [--n_cpus N_CPUS] [--hidden_nf HIDDEN_NF] [--prot_dist_threshold PROT_DIST_THRESHOLD] [--intra_cutoff INTRA_CUTOFF] [--inter_cutoff INTER_CUTOFF]
                [--random_state RANDOM_STATE] [--lr LR] [--loss_type {no_avg,avg_over_graph,avg_over_mol}] [--loss_function {BCEWithLogitsLoss,BCELoss}] [--act_function {SiLU}] [--n_layers N_LAYERS] [--use_wandb]
                [--verbose] [--use_lr_scheduler] [--lr_scheduler_type {ReduceLROnPlateau,Linear}] [--data_split_file DATA_SPLIT_FILE]

options:
  -h, --help            show this help message and exit
  --data_json DATA_JSON
                        dict containing codes assigned to unique ligands and filenames (not paths) for mols (sdf/mol) and pdb files {lig_codes: [], mol_files: [], pdb_files: []}
  --pdb_dir PDB_DIR     path where the pdb fpaths are located
  --precursor_dir PRECURSOR_DIR
                        path where the mol fpaths are located
  --processed_dir PROCESSED_DIR
                        dir to save processed data
  --model_dir MODEL_DIR
                        dir to save outputs from the model
  --run_name RUN_NAME   unique name for saving files and wandb project name
  --n_epochs N_EPOCHS
  --patience PATIENCE
  --batch_size BATCH_SIZE
  --test_size TEST_SIZE
                        if no specified data split, creates test set from data randomly
  --n_cpus N_CPUS
  --hidden_nf HIDDEN_NF
  --prot_dist_threshold PROT_DIST_THRESHOLD
                        threshold for protein atoms to consider according to distance from ligand atoms
  --intra_cutoff INTRA_CUTOFF
                        distance threshold for intra-mol edges
  --inter_cutoff INTER_CUTOFF
                        distnace threshold for inter-mol edges
  --random_state RANDOM_STATE
                        random state for train/test split (if used)
  --lr LR               learning rate
  --loss_type {no_avg,avg_over_graph,avg_over_mol}
                        how to calculate loss
  --loss_function {BCEWithLogitsLoss,BCELoss}
  --act_function {SiLU}
  --n_layers N_LAYERS   number of hidden layers
  --use_wandb
  --verbose
  --use_lr_scheduler    whether to use an LR scheduler
  --lr_scheduler_type {ReduceLROnPlateau,Linear}
  --data_split_file DATA_SPLIT_FILE
                        json file with dictionary containing idxs of data split {train: [...], test: ..., validation: ...}
```
Things you may want to change:

- loss_type can be calculated in different ways; I used avg_over_mol to calculate based only on ligand nodes and take the avg for each molecule (rather than across whole batch or graph incl protein nodes)
- may want to change how positive weight applied to loss function (currently done by calculating proportion of positive labels in the train dataset, can specify to do over all ligand nodes only, depending on how loss is averaged)
- LR scheduler -- may want to experiment more with this?
- Should separate out test set evaluation from train.py (i.e. I selected best model based on val set and then separately used the that model to eval on the test set)

I have also written functions (in `utils/viz.py`) to output pdb files for the predictions by adding B factor values representing the probabilities.
You can visualize this in PyMol with `spectrum b, blue_red, minimum=0, maximum=1`.