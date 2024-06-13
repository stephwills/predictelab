
import os

import torch
from src.dataset.preprocessing import convert_to_graph
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ElabDataset(Dataset):

    def __init__(self, lig_codes=None, mol_files=None, pdb_files=None, list_of_vectors=None, prot_dist_threshold=8, intra_cutoff=2, inter_cutoff=10,
                 mol_file_suffix='.mol', pdb_file_suffix='_receptor.pdb', verbose=True, processed_dir=None, save_processed_files=False):
        """

        :param lig_codes:
        :param mol_files:
        :param pdb_files:
        :param prot_dist_threshold:
        :param intra_cutoff:
        :param inter_cutoff:
        :param mol_file_suffix:
        :param pdb_file_suffix:
        :param verbose:
        :param processed_dir:
        :param save_processed_files:
        """
        self.lig_codes = lig_codes

        # if data not processed already
        self.mol_files = mol_files
        self.pdb_files = pdb_files
        # if vectors are not supplier, assume stored as molprop
        self.list_of_vectors = list_of_vectors
        self.mol_file_suffix = mol_file_suffix
        self.pdb_file_suffix = pdb_file_suffix

        # to save pre-processed files
        self.processed_dir = processed_dir
        self.save_processed_files = save_processed_files
        self.processed_data = None

        # thresholds
        self.prot_dist_threshold = prot_dist_threshold
        self.intra_cutoff = intra_cutoff
        self.inter_cutoff = inter_cutoff

        # misc
        self.verbose = verbose

        # load existing processed data or run processing
        self._process_data()

    def _load_processed_files(self, files):
        """
        Load the processed data to self.processed_data from existing .pt files

        :param files: .pt files to load data dict from
        :return:
        """
        self.processed_data = []

        for file in files:
            d = torch.load(file)
            self.processed_data.append(d)

    def _process_data(self):
        """
        Check if data has already been processed (saved in self.processed_dir); otherwise process the pdb and mol files

        :return:
        """
        # check if data has already been processed
        if self.processed_dir:
            # can either just read all files or use pre-specified ligand codes
            self.pt_files = [os.path.join(self.processed_dir, file) for file in os.listdir(self.processed_dir)]

            if len(self.pt_files) > 0:
                if self.lig_codes:  # if lig codes are specified, check files exist
                    filt_lig_codes = []
                    filt_files = []
                    files = [os.path.join(self.processed_dir, f"{lig_code}.pt") for lig_code in self.lig_codes]
                    for file, lig_code in zip(files, self.lig_codes):
                        if os.path.exists(file):
                            filt_lig_codes.append(lig_code)
                            filt_files.append(file)
                    self.lig_codes = filt_lig_codes
                    self.pt_files = filt_files
                else:
                    self.lig_codes = [os.path.basename(pt_file).replace('.pt', '') for pt_file in self.pt_files]

                self._load_processed_files(self.pt_files)
                if self.verbose:
                    print(len(self.lig_codes), 'complexes loaded')
                return

        # if not processed already, then process pdb and mol files
        filt_lig_codes = []
        filt_pdb_files = []
        filt_mol_files = []

        if self.lig_codes:
            for pdb_file, mol_file, lig_code in zip(self.pdb_files, self.mol_files, self.lig_codes):
                if os.path.exists(pdb_file) and os.path.exists(mol_file):
                    filt_pdb_files.append(pdb_file)
                    filt_mol_files.append(mol_file)
                    filt_lig_codes.append(lig_code)
        else:
            for i, (pdb_file, mol_file) in enumerate(zip(self.pdb_files, self.mol_files)):
                if os.path.exists(pdb_file) and os.path.exists(mol_file):
                    lig_code = f'lig-{i}'
                    filt_pdb_files.append(pdb_file)
                    filt_mol_files.append(mol_file)
                    filt_lig_codes.append(lig_code)

        self.lig_codes = filt_lig_codes
        self.pdb_files = filt_pdb_files
        self.mol_files = filt_mol_files

        self._run_processing()

        if self.verbose:
            print(len(self.lig_codes), 'complexes loaded')

    def _run_processing(self):
        """
        Run the actual processing for pdb and mol files; optional to save into processed dir

        :return:
        """
        if self.save_processed_files:
            self.pt_files = []

        self.processed_data = []

        if self.list_of_vectors:
            for pdb_file, mol_file, lig_code, vectors in zip(self.pdb_files, self.mol_files, self.lig_codes, self.list_of_vectors):
                h, y, x, edge_index, edge_attr = convert_to_graph(mol_file, pdb_file, self.prot_dist_threshold, self.intra_cutoff, self.inter_cutoff, vectors=vectors, vectors_are_molprop=False)
                data_dict = {'node_feats': h,
                                'labels': y,
                                'pos': x,
                                'edges': edge_index,
                                'edge_feats': edge_attr}
                self.processed_data.append(data_dict)
                if self.save_processed_files and self.processed_dir:
                    pt_file = os.path.join(self.processed_dir, f"{lig_code}.pt")
                    torch.save(data_dict, pt_file)
                    self.pt_files.append(pt_file)
        else:
            for pdb_file, mol_file, lig_code in zip(self.pdb_files, self.mol_files, self.lig_codes):
                h, y, x, edge_index, edge_attr = convert_to_graph(mol_file, pdb_file, self.prot_dist_threshold, self.intra_cutoff, self.inter_cutoff, vectors=None, vectors_are_molprop=True)
                data_dict = {'node_feats': h,
                                'labels': y,
                                'pos': x,
                                'edges': edge_index,
                                'edge_feats': edge_attr}
                self.processed_data.append(data_dict)
                if self.save_processed_files and self.processed_dir:
                    pt_file = os.path.join(self.processed_dir, f"{lig_code}.pt")
                    torch.save(data_dict, pt_file)
                    self.pt_files.append(pt_file)

    def __len__(self):
        """
        Get number of data points

        :return:
        """
        return len(self.lig_codes)

    def __getitem__(self, idx):
        """
        Get item by accessing self.processed_data

        :param idx:
        :return:
        """
        if not self.processed_data:
            pdb_file, mol_file = self.pdb_files[idx], self.mol_files[idx]
            if self.list_of_vectors:
                vectors = self.list_of_vectors[idx]
                h, y, x, edge_index, edge_attr = convert_to_graph(mol_file, pdb_file, self.prot_dist_threshold, self.intra_cutoff, self.inter_cutoff,
                                                                  vectors=vectors, vectors_are_molprop=False)
            else:
                h, y, x, edge_index, edge_attr = convert_to_graph(mol_file, pdb_file, self.prot_dist_threshold, self.intra_cutoff, self.inter_cutoff,
                                                                  vectors=None, vectors_are_molprop=True)
            data = Data(h, edge_index, edge_attr, y, x)
            return data

        else:
            data_dict = self.processed_data[idx]
            h, y, x, edge_index, edge_attr = data_dict['node_feats'], data_dict['labels'], data_dict['pos'], \
                                             data_dict['edges'], data_dict['edge_feats']
            data = Data(h, edge_index, edge_attr, y, x)
            return data
