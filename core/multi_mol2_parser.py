import os
import tempfile
import re
from typing import List, Tuple, Dict
from rdkit import Chem
import numpy as np

def split_multi_mol2_file(multi_mol2_path: str, output_dir: str = None) -> List[Tuple[str, str]]:
    """
    将包含多个分子的MOL2文件拆分为多个单独的文件
    
    参数:
    - multi_mol2_path: 包含多个分子的MOL2文件路径
    - output_dir: 输出目录，如果为None则创建临时目录
    
    返回:
    - List[Tuple[str, str]]: 每个元素为(分子名称, 文件路径)的列表
    """
    if not os.path.exists(multi_mol2_path):
        raise FileNotFoundError(f"文件不存在: {multi_mol2_path}")
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mol2_split_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(multi_mol2_path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式分割多个分子
    # 查找所有MOLECULE段的开始位置
    molecule_starts = [m.start() for m in re.finditer(r'@<TRIPOS>MOLECULE', content)]
    
    molecules = []
    
    for i, start_pos in enumerate(molecule_starts):
        if i < len(molecule_starts) - 1:
            # 不是最后一个分子
            molecule_content = content[start_pos:molecule_starts[i+1]]
        else:
            # 最后一个分子
            molecule_content = content[start_pos:]
        
        # 提取分子名称（MOLECULE段后的第一行）
        lines = molecule_content.strip().split('\n')
        if len(lines) > 1:
            molecule_name = lines[1].strip()
            
            # 清理分子名称，移除特殊字符
            molecule_name = re.sub(r'[^\w\-_]', '_', molecule_name)
            
            # 如果名称为空，使用索引
            if not molecule_name or molecule_name.isspace():
                molecule_name = f"molecule_{i+1:04d}"
        else:
            molecule_name = f"molecule_{i+1:04d}"
        
        # 保存为单独的文件
        output_path = os.path.join(output_dir, f"{molecule_name}.mol2")
        
        with open(output_path, 'w') as f:
            f.write(molecule_content)
        
        molecules.append((molecule_name, output_path))
    
    return molecules

def parse_multi_mol2_metadata(multi_mol2_path: str) -> List[Dict]:
    """
    解析多构象MOL2文件的元数据，不实际拆分文件
    
    参数:
    - multi_mol2_path: 包含多个分子的MOL2文件路径
    
    返回:
    - List[Dict]: 每个分子的元数据列表
    """
    if not os.path.exists(multi_mol2_path):
        raise FileNotFoundError(f"文件不存在: {multi_mol2_path}")
    
    with open(multi_mol2_path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式分割多个分子
    molecule_starts = [m.start() for m in re.finditer(r'@<TRIPOS>MOLECULE', content)]
    
    molecules_metadata = []
    
    for i, start_pos in enumerate(molecule_starts):
        if i < len(molecule_starts) - 1:
            molecule_content = content[start_pos:molecule_starts[i+1]]
        else:
            molecule_content = content[start_pos:]
        
        lines = molecule_content.strip().split('\n')
        
        if len(lines) < 2:
            continue
        
        # 提取分子名称
        molecule_name = lines[1].strip()
        
        # 尝试从注释行中提取更多信息
        comment_lines = []
        for line in lines:
            if line.startswith('#'):
                comment_lines.append(line.strip())
        
        metadata = {
            'index': i + 1,
            'name': molecule_name,
            'content': molecule_content,
            'comments': comment_lines,
            'total_atoms': 0,
            'total_bonds': 0
        }
        
        # 尝试从MOLECULE行中提取原子数和键数
        if len(lines) >= 3:
            # 第三行通常包含原子数、键数等信息
            info_line = lines[2].strip()
            parts = info_line.split()
            if len(parts) >= 2:
                try:
                    metadata['total_atoms'] = int(parts[0])
                    metadata['total_bonds'] = int(parts[1])
                except:
                    pass
        
        molecules_metadata.append(metadata)
    
    return molecules_metadata

def read_multi_mol2_as_list(multi_mol2_path: str) -> List[Chem.rdchem.Mol]:
    """
    将多构象MOL2文件读取为RDKit分子列表
    
    参数:
    - multi_mol2_path: 包含多个分子的MOL2文件路径
    
    返回:
    - List[Chem.rdchem.Mol]: RDKit分子列表
    """
    
    
    molecules_metadata = parse_multi_mol2_metadata(multi_mol2_path)
    molecules = []
    
    for metadata in molecules_metadata:
        try:
            mol = Chem.MolFromMol2Block(metadata['content'], sanitize=False)
            if mol:
                Chem.SanitizeMol(mol)
                molecules.append(mol)
        except Exception as e:
            print(f"无法解析分子 {metadata['name']}: {e}")
            continue
    
    return molecules

def extract_molecule_from_multi_mol2(multi_mol2_path: str, molecule_index: int) -> str:
    """
    从多构象MOL2文件中提取单个分子
    
    参数:
    - multi_mol2_path: 包含多个分子的MOL2文件路径
    - molecule_index: 分子索引（从1开始）
    
    返回:
    - str: 分子内容字符串
    """
    molecules_metadata = parse_multi_mol2_metadata(multi_mol2_path)
    
    if molecule_index < 1 or molecule_index > len(molecules_metadata):
        raise IndexError(f"分子索引超出范围: {molecule_index} (总共 {len(molecules_metadata)} 个分子)")
    
    return molecules_metadata[molecule_index-1]['content']