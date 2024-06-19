import os

def generate_fnames_from_dir(dir):
    lig_codes = [code for code in os.listdir(dir)]
    mol_files = [os.path.join(dir, code, f"{code}.sdf") for code in lig_codes]
    receptor_files = [os.path.join(dir, code, f"{code}_receptor.pdb") for code in lig_codes]
    return lig_codes, mol_files, receptor_files

def get_pos_weight(data):
    y = data.y
    pos_weight = (y==0.).sum()/y.sum()
    return pos_weight
