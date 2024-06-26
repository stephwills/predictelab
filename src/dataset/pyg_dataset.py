
import os

from src.dataset.preprocessing import convert_to_graph
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class ElabDataset(InMemoryDataset):

    def __init__(self, root, lig_codes, mol_files, pdb_files, list_of_vectors=None, processed_dir=None, save_processed_files=True,
                 prot_dist_threshold=8, intra_cutoff=2, inter_cutoff=10, verbose=True, transform=None, pre_transform=None, pre_filter=None):

        self.lig_codes = lig_codes
        self.mol_files = mol_files
        self.pdb_files = pdb_files
        self.list_of_vectors = list_of_vectors

        self.proc_dir = processed_dir
        self.save_processed_files = save_processed_files
        self.processed_data = []
        self.pt_files = []
        self.verbose = verbose

        self.prot_dist_threshold = prot_dist_threshold
        self.intra_cutoff = intra_cutoff
        self.inter_cutoff = inter_cutoff

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.mol_files[0]

    @property
    def processed_file_names(self):
        return [os.path.join(self.proc_dir, i) for i in os.listdir(self.proc_dir)]

    def download(self):
        pass

    def process(self):
        # check files are present
        filt_lig_codes = []
        filt_pdb_files = []
        filt_mol_files = []

        for pdb_file, mol_file, lig_code in zip(self.pdb_files, self.mol_files, self.lig_codes):
            if os.path.exists(pdb_file) and os.path.exists(mol_file):
                filt_pdb_files.append(pdb_file)
                filt_mol_files.append(mol_file)
                filt_lig_codes.append(lig_code)

        self.lig_codes = filt_lig_codes
        self.pdb_files = filt_pdb_files
        self.mol_files = filt_mol_files

        if self.list_of_vectors:
            for pdb_file, mol_file, lig_code, vectors in zip(self.pdb_files, self.mol_files, self.lig_codes, self.list_of_vectors):
                h, y, x, edge_index, edge_attr, lig_mask, atom_idxs = convert_to_graph(mol_file, pdb_file, self.prot_dist_threshold, self.intra_cutoff, self.inter_cutoff,
                                                                  vectors=vectors, vectors_are_molprop=False, return_node_info=True)
                data_dict = Data(h, edge_index, edge_attr, y, x)
                data_dict.lig_code = lig_code
                data_dict.lig_mask = lig_mask
                data_dict.atom_idxs = atom_idxs
                self.processed_data.append(data_dict)

        else:
            for pdb_file, mol_file, lig_code in zip(self.pdb_files, self.mol_files, self.lig_codes):
                h, y, x, edge_index, edge_attr, lig_mask, atom_idxs = convert_to_graph(mol_file, pdb_file, self.prot_dist_threshold, self.intra_cutoff, self.inter_cutoff,
                                                                  vectors=None, vectors_are_molprop=True, return_node_info=True)
                data_dict = Data(h, edge_index, edge_attr, y, x)
                data_dict.lig_code = lig_code
                data_dict.lig_mask = lig_mask
                data_dict.atom_idxs = atom_idxs
                self.processed_data.append(data_dict)

        self.save(self.processed_data, os.path.join(self.proc_dir, 'proc.pt'))
        if self.verbose:
            print(len(self.lig_codes), 'complexes loaded')

