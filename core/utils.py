import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
from pydockrmsd.dockrmsd import PyDockRMSD

def generate_3d_dist(mtx_1, mtx_2):
    """计算3D距离矩阵"""
    n, N, C = mtx_1.size()
    n, M, _ = mtx_2.size()
    dist = -2 * torch.matmul(mtx_1, mtx_2.permute(0, 2, 1))
    dist += torch.sum(mtx_1 ** 2, -1).view(-1, N, 1)
    dist += torch.sum(mtx_2 ** 2, -1).view(-1, 1, M)
    
    dist = (dist >= 0) * dist
    dist = torch.sqrt(dist)
    return dist

# 残基修正字典
correct_residue_dict = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HIZ": "HIS", "HIY": "HIS",
    "CYX": "CYS", "CYM": "CYS", "CYT": "CYS", "MEU": "MET", "LEV": "LEU",
    "ASQ": "ASP", "ASH": "ASP", "DID": "ASP", "DIC": "ASP", "GLZ": "GLY",
    "GLV": "GLU", "GLH": "GLU", "GLM": "GLU", "ASZ": "ASN", "ASM": "ASN",
    "GLO": "GLN", "SEM": "SER", "TYM": "TYR", "ALB": "ALA"
}

rec_defined_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
                       'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
                       'GLU', 'LYS', 'ARG', 'HIS', "OTH"]

HETATM_list = ["WA", "HEM", "NAD", "NAP", "UMP", "MG", "SAM", "ADP", "FAD", 
               "CA", "ZN", "FMN", "CA", "NDP", "TPO", "LLP"]

def calculate_dockrmsd(mol2file1, mol2file2):
    """计算两个mol2文件之间的dockrmsd"""
    try:
        dockrmsd = PyDockRMSD(mol2file1, mol2file2)
        d_rmsd = dockrmsd.rmsd
        return d_rmsd
    except Exception as e:
        print(f"计算dockrmsd时出错: {e}")
        return float('inf')

def convert_mol2_to_mol2_dH(ref_lig_fpath, ref_lig_dH_fpath):
    """对mol2文件进行去H操作"""
    subprocess.run(['obabel', ref_lig_fpath, '-O', ref_lig_dH_fpath, '-d'], 
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(ref_lig_dH_fpath, 'r') as infile:
        lines = infile.readlines()

    with open(ref_lig_dH_fpath, 'w') as outfile:
        skip = False
        for line in lines:
            if line.startswith("@<TRIPOS>UNITY_ATOM_ATTR"):
                skip = True
            elif line.startswith("@<TRIPOS>"):
                skip = False
            if not skip:
                outfile.write(line)

def collate_fn(batch):
    """批次数据整理函数"""
    max_n_node = max(item['pos'].size(0) for item in batch)
    batch_size = len(batch)

    padded_pos = torch.zeros(batch_size, max_n_node, 3)
    padded_atoms = torch.zeros(batch_size, max_n_node).long()
    padded_tags = torch.zeros(batch_size, max_n_node).long()
    padding_mask = torch.ones(batch_size, max_n_node).bool()
    
    d_rmsds = []
    sample_pose_ids = []

    for i, item in enumerate(batch):
        n_node = item['pos'].size(0)
        
        padded_pos[i, :n_node] = item['pos']
        padded_atoms[i, :n_node] = item['atoms']
        padded_tags[i, :n_node] = item['tags']
        
        padding_mask[i, :n_node] = False
        d_rmsds.append(item.get('d_rmsd', 0.0))
        sample_pose_ids.append(item.get('sample_pose_ids', f'batch_{i}'))

    return {
        'pos': padded_pos,
        'atoms': padded_atoms,
        'tags': padded_tags,
        'padding_mask': padding_mask,
        'sample_pose_ids': sample_pose_ids
    }

def save_optimized_structure(input_mol2, optimized_coords, output_mol2):
    """保存优化后的结构到mol2文件"""
    try:
        with open(input_mol2, 'r') as f:
            lines = f.readlines()
        
        # 找到坐标部分的开始和结束
        coord_start = None
        coord_end = None
        for i, line in enumerate(lines):
            if '@<TRIPOS>ATOM' in line:
                coord_start = i + 1
            elif coord_start is not None and ('@<TRIPOS>' in line and i > coord_start):
                coord_end = i
                break
        
        if coord_start is None:
            raise ValueError("无法找到原子坐标部分")
        
        if coord_end is None:
            coord_end = len(lines)
        
        # 替换坐标部分
        new_lines = lines[:coord_start]
        
        for i in range(coord_start, coord_end):
            if i < len(lines):
                line = lines[i].strip()
                if line and not line.startswith('@'):
                    parts = line.split()
                    if len(parts) >= 6:
                        atom_idx = i - coord_start
                        if atom_idx < len(optimized_coords):
                            new_coord = optimized_coords[atom_idx]
                            new_line = f"{parts[0]:>7} {parts[1]:<8} {new_coord[0]:>9.4f} {new_coord[1]:>9.4f} {new_coord[2]:>9.4f} {parts[5]}"
                            if len(parts) > 6:
                                new_line += " " + " ".join(parts[6:])
                            new_lines.append(new_line + "\n")
                        else:
                            new_lines.append(lines[i])
                    else:
                        new_lines.append(lines[i])
                else:
                    new_lines.append(lines[i])
            else:
                break
        
        new_lines.extend(lines[coord_end:])
        
        with open(output_mol2, 'w') as f:
            f.writelines(new_lines)
        
        return True
        
    except Exception as e:
        print(f"保存优化结构时出错: {e}")
        return False