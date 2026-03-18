import os
import torch
import numpy as np
import re
from rdkit import Chem
from collections import OrderedDict
from .utils import generate_3d_dist, correct_residue_dict, rec_defined_residues, HETATM_list

# 定义与您提供的代码一致的元素类别映射
element_category = OrderedDict({
    'C': 'C', 'N': 'N', 'O': 'O', 'S': 'S',
    'F': 'Hal', 'Cl': 'Hal', 'Br': 'Hal', 'I': 'Hal',
    'Du': 'Du',
})

# 标准残基
standard_residues = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'OTH', 'LIG'
]

# 编码映射
encoding_map = OrderedDict()
current_idx = 0
for residue in standard_residues:
    for category in element_category.values():
        if (category, residue) not in encoding_map:
            encoding_map[(category, residue)] = current_idx
            current_idx += 1

def encode_atom_type(element: str, residue: str) -> int:
    """原子类型编码"""
    element = element.upper() if element != 'Du' else element
    atom_category = element_category.get(element, 'Du')
    
    residue = residue.upper()
    if residue not in standard_residues:
        residue = 'OTH'
    
    return encoding_map.get((atom_category, residue), 0)

def encoder_atom_res(ele_list, residue_list):
    """原子类型编码器"""
    encoded_results = []
    for i in range(len(ele_list)):
        encoded = encode_atom_type(ele_list[i], residue_list[i])
        encoded_results.append(encoded)
    return encoded_results

def extract_ligand_features(mol2_path):
    """从mol2文件提取配体特征"""

    
    # 读取mol2文件
    with open(mol2_path, 'r') as f:
        content = f.read()
    
    # 修复mol2格式
    fixed_content = []
    in_molecule = False
    line_counter = 0
    
    for line in content.split('\n'):
        if line.startswith('@<TRIPOS>MOLECULE'):
            in_molecule = True
            line_counter = 0
            fixed_content.append(line)
        elif in_molecule:
            line_counter += 1
            if line_counter == 1:
                fixed_content.append(line.strip())
            elif line_counter == 2:
                cleaned_line = re.sub(r'^\s+', '', line)
                fixed_content.append(cleaned_line)
            else:
                fixed_content.append(line)
            
            if line_counter >= 4:
                in_molecule = False
        else:
            fixed_content.append(line)
    
    mol_block = '\n'.join(fixed_content)
    mol = Chem.MolFromMol2Block(mol_block, sanitize=False)
    if mol:
        Chem.SanitizeMol(mol)
    
    if mol is None:
        raise ValueError(f"无法读取配体文件: {mol2_path}")
    
    # 提取配体特征
    lig_heavy_xyz = []
    lig_atom_type = []
    
    if mol:
        coords = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'H':  # 只考虑重原子
                lig_heavy_xyz.append(coords[atom.GetIdx()])
                lig_atom_type.append(atom.GetSymbol())
    
    return np.array(lig_heavy_xyz, dtype=np.float32), lig_atom_type

class ReceptorLoader:

    def __init__(self, rec_fpath: str, ligand_xyz: torch.Tensor, clip_cutoff: float = 8.0):
        self.rec_fpath = rec_fpath
        self.ligand_xyz = ligand_xyz
        self.clip_cutoff = clip_cutoff
        
    def load(self):
        with open(self.rec_fpath) as f:
            lines = [line.strip() for line in f.readlines() if line[:4] == "ATOM"]

        all_atom_xyz_list = []
        all_elements = []
        all_atom_residues = []
        all_pdb_types = []
        all_resid_atom_indices = []
        resid_symbol_pool = []
        residues = []
        temp_res_xyz = []
        temp_indices_list = []
        temp_pdb_types_list = []
        
        atom_idx = -1
        res_idx = -1
        
        for num, line in enumerate(lines):
            resid_symbol = line[17:27].strip()
            res = line[17:20].strip()
            if res in HETATM_list or res[:2] in HETATM_list:
                continue
            if not res in rec_defined_residues:
                try:
                    res = correct_residue_dict[res]
                except KeyError:
                    res = "OTH"
            ele = line.split()[-1]
            if not ele in ["H", "C", "N", "O", "S"]:
                ele = "DU"

            atom_idx += 1
            pdb_type = line[11:17].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atom_xyz = np.c_[x, y, z]

            all_elements.append(ele)
            all_pdb_types.append(pdb_type)
            all_atom_residues.append(res)
            all_atom_xyz_list.append(atom_xyz)

            if ele == "H":
                continue

            if num == 0:
                resid_symbol_pool.append(resid_symbol)
                residues.append(res)
                temp_res_xyz.append(atom_xyz)
                temp_pdb_types_list.append(pdb_type)
                temp_indices_list.append(atom_idx)
            else:
                if resid_symbol != resid_symbol_pool[-1]:
                    res_idx += 1
                    all_resid_atom_indices.append(temp_indices_list)
                    
                    resid_symbol_pool.append(resid_symbol)
                    residues.append(res)
                    temp_res_xyz = [atom_xyz]
                    temp_pdb_types_list = [pdb_type]
                    temp_indices_list = [atom_idx]
                else:
                    temp_res_xyz.append(atom_xyz)
                    temp_pdb_types_list.append(pdb_type)
                    temp_indices_list.append(atom_idx)
        
        # 处理最后残基
        res_idx += 1
        all_resid_atom_indices.append(temp_indices_list)
        
        all_atom_xyz_tensor = torch.from_numpy(np.concatenate(all_atom_xyz_list, axis=0)).to(torch.float32)
        
        # 裁剪口袋
        clip_ha_indices = []
        clip_rec_ele = []
        clip_rec_ha_residues = []
        clip_rec_ha_xyz_list = []
        
        for res_idx, atom_indices in enumerate(all_resid_atom_indices):
            res_atoms_xyz = all_atom_xyz_tensor[atom_indices]
            
            # 计算与配体的距离
            distances = torch.cdist(res_atoms_xyz.unsqueeze(0), self.ligand_xyz.unsqueeze(0)).squeeze(0)
            min_dist = torch.min(distances)
            
            if min_dist <= self.clip_cutoff:
                for atom_idx in atom_indices:
                    clip_ha_indices.append(atom_idx)
                    clip_rec_ele.append(all_elements[atom_idx])
                    clip_rec_ha_residues.append(all_atom_residues[atom_idx])
                    clip_rec_ha_xyz_list.append(all_atom_xyz_tensor[atom_idx].unsqueeze(0))
        
        if clip_rec_ha_xyz_list:
            clip_rec_ha_xyz_tensor = torch.cat(clip_rec_ha_xyz_list, axis=0)
        else:
            clip_rec_ha_xyz_tensor = torch.empty((0, 3), dtype=torch.float32)
        
        return clip_rec_ha_xyz_tensor.numpy(), clip_rec_ele, clip_rec_ha_residues

def extract_complex_features(protein_pdb: str, ligand_mol2: str):
    """
    从原始PDB和MOL2文件提取复合物特征
    
    参数:
    - protein_pdb: 蛋白质PDB文件路径
    - ligand_mol2: 配体MOL2文件路径
    
    返回:
    - features: 包含pos, atoms, tags的字典
    """
    # 1. 提取配体特征
    ligand_coords, ligand_elements = extract_ligand_features(ligand_mol2)
    if ligand_coords is None or len(ligand_coords) == 0:
        raise ValueError("未找到配体重原子")
    
    # 2. 编码配体原子类型
    encode_lig_atom = encoder_atom_res(ligand_elements, ['LIG'] * len(ligand_elements))
    
    all_xyz = ligand_coords
    all_atom_types = encode_lig_atom
    tags = [1] * len(ligand_coords)  # 配体原子标签为1
    
    # 3. 如果提供了受体，提取口袋特征
    if protein_pdb and os.path.exists(protein_pdb):
        try:
            ligand_coords_tensor = torch.tensor(ligand_coords, dtype=torch.float32)
            receptor_loader = ReceptorLoader(protein_pdb, ligand_coords_tensor)
            pocket_ha_xyz_list, pocket_ele, pocket_ha_residues = receptor_loader.load()
            
            if len(pocket_ha_xyz_list) > 0:
                encode_pocket_atom = encoder_atom_res(pocket_ele, pocket_ha_residues)
                
                protein_tags = [0] * len(pocket_ha_xyz_list)
                ligand_tags = [1] * len(ligand_coords)
                tags = ligand_tags + protein_tags
                
                all_xyz = np.concatenate([ligand_coords, pocket_ha_xyz_list])
                all_atom_types = encode_lig_atom + encode_pocket_atom
        except Exception as e:
            print(f"受体特征提取失败: {e}")
            print("  将仅使用配体特征")
    
    # 转换为张量
    features = {
        'pos': torch.tensor(all_xyz, dtype=torch.float32),
        'atoms': torch.tensor(all_atom_types, dtype=torch.long),
        'tags': torch.tensor(tags, dtype=torch.long),
    }
    
    return features