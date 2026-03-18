import os
import torch
import pickle
import numpy as np
import tempfile
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
from core.feature_extractor import extract_complex_features
from core.multi_mol2_parser import split_multi_mol2_file, parse_multi_mol2_metadata

class Graphormer3DDataset(Dataset):
    """可以从原始PDB/MOL2文件或预提取的pickle特征文件加载数据。"""
    
    def __init__(self, index_path: Optional[str] = None, feature_dir: Optional[str] = None,
                 protein_pdb_list: Optional[List[str]] = None, 
                 ligand_mol2_list: Optional[List[str]] = None,
                 is_multi_mol2: bool = False,
                 temp_dir: Optional[str] = None):
        """
        参数:
        - is_multi_mol2: 是否为多构象MOL2文件
        - temp_dir: 临时目录用于存储拆分的分子文件
        """
        self.samples = []
        self.mode = None
        self.temp_dir = temp_dir
        self.split_files = []  # 存储拆分后的临时文件
        
        if index_path and feature_dir:
            self.mode = 'pickle'
            self._load_from_pickle(index_path, feature_dir)
        elif ligand_mol2_list is not None:
            self.mode = 'raw'
            if is_multi_mol2 and len(ligand_mol2_list) == 1:
                # 多构象MOL2文件处理
                self._load_from_multi_mol2(
                    ligand_mol2_list[0], 
                    protein_pdb_list[0] if protein_pdb_list else None
                )
            else:
                # 原有单分子处理逻辑
                self._load_from_raw(protein_pdb_list, ligand_mol2_list)
        else:
            raise ValueError("必须提供有效的初始化参数组合。")
    
    def _load_from_multi_mol2(self, multi_mol2_path: str, protein_pdb: Optional[str]):
        """从多构象MOL2文件加载数据"""
        print(f"处理多构象MOL2文件: {os.path.basename(multi_mol2_path)}")
        
        # 解析多构象文件的元数据
        molecules_metadata = parse_multi_mol2_metadata(multi_mol2_path)
        print(f"  发现 {len(molecules_metadata)} 个构象")
        
        # 拆分多构象文件
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="graphormer_multi_mol2_")
        
        split_results = split_multi_mol2_file(multi_mol2_path, self.temp_dir)
        
        # 存储临时文件路径，供清理使用
        self.split_files = [path for _, path in split_results]
        
        # 为每个构象创建样本
        for i, (mol_name, mol_path) in enumerate(split_results):
            sample_id = f"multi_{i:04d}"
            pose_id = mol_name
            
            self.samples.append({
                'protein_pdb': protein_pdb,
                'ligand_mol2': mol_path,
                'sample_id': sample_id,
                'pose_id': pose_id,
                'original_mol2': multi_mol2_path,
                'molecule_index': i + 1
            })
        
        print(f"  已准备 {len(self.samples)} 个样本用于处理")
    
    def _load_from_raw(self, protein_pdb_list, ligand_mol2_list):

        n_ligands = len(ligand_mol2_list)
        n_proteins = len(protein_pdb_list) if protein_pdb_list else 0
        
        
        if n_proteins == 1 and n_ligands > 1:
            protein_pdb_list = [protein_pdb_list[0]] * n_ligands
        elif n_proteins != n_ligands:
            raise ValueError(f"蛋白质文件数({n_proteins})与配体文件数({n_ligands})不匹配。")
        
        for i, (prot, lig) in enumerate(zip(protein_pdb_list, ligand_mol2_list)):
            sample_id = f"sample_{i:04d}"
            pose_id = os.path.splitext(os.path.basename(lig))[0]
            self.samples.append({
                'protein_pdb': prot,
                'ligand_mol2': lig,
                'sample_id': sample_id,
                'pose_id': pose_id
            })
    
    def _load_from_pickle(self, index_path, feature_dir):
        """从pickle文件加载"""
        with open(index_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    sample_id, pose_id = parts[0], parts[1]
                    filename = os.path.join(sample_id, f"{pose_id}.pkl")
                    data_path = os.path.join(feature_dir, filename)
                    if os.path.exists(data_path):
                        self.samples.append((data_path, sample_id, pose_id))
                    else:
                        print(f"警告: 特征文件不存在，已跳过 - {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode == 'pickle':
            return self._getitem_pickle(idx)
        else:
            return self._getitem_raw(idx)
    
    def _getitem_pickle(self, idx):
        """从pickle文件获取数据"""
        file_path, sample_id, pose_id = self.samples[idx]
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"加载pickle文件失败 {file_path}: {e}")
            raise
        
        return {
            'pos': torch.tensor(data['pos'], dtype=torch.float32),
            'atoms': torch.tensor(data['atoms'], dtype=torch.long),
            'tags': torch.tensor(data['tags'], dtype=torch.long),
            'sample_pose_ids': f"{sample_id} {pose_id}"
        }
    
    def _getitem_raw(self, idx):
        """从原始文件获取数据"""
        sample_info = self.samples[idx]
        prot_path = sample_info['protein_pdb']
        lig_path = sample_info['ligand_mol2']
        sample_id = sample_info['sample_id']
        pose_id = sample_info['pose_id']
        
        try:
            features = extract_complex_features(prot_path, lig_path)
        except Exception as e:
            print(f"处理文件失败 受体:{prot_path}, 配体:{lig_path}: {e}")
            raise
        
        return {
            'pos': features['pos'],
            'atoms': features['atoms'],
            'tags': features['tags'],
            'sample_pose_ids': f"{sample_id} {pose_id}"
        }
    
    def cleanup_temp_files(self):
        """清理临时拆分的文件"""
        if self.split_files:
            import shutil
            for file_path in self.split_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
            
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except:
                    pass
    @classmethod
    def from_raw_files(cls, protein_pdb_list, ligand_mol2_list):

        return cls(
            protein_pdb_list=protein_pdb_list,
            ligand_mol2_list=ligand_mol2_list

        )
    @classmethod
    def from_multi_mol2_file(cls, protein_pdb: str, multi_mol2_path: str, 
                            temp_dir: Optional[str] = None):

        return cls(
            protein_pdb_list=[protein_pdb] if protein_pdb else [None],
            ligand_mol2_list=[multi_mol2_path],
            is_multi_mol2=True,
            temp_dir=temp_dir
        )
