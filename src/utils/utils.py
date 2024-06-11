import os

def generate_fnames_from_dir(dir):
    lig_codes = [code for code in os.listdir(dir)]
    mol_files = [os.path.join(dir, code, f"{code}.mol") for code in lig_codes]
    receptor_files = [os.path.join(dir, code, f"{code}_receptor.pdb") for code in lig_codes]
    return lig_codes, mol_files, receptor_files
