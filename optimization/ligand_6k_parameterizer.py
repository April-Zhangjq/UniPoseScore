import torch
import numpy as np
from scipy.optimize import minimize
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.optim as optim

class Ligand6KParameterizer:
    """将配体表示为6+k个参数的工具类"""
    
    def __init__(self, mol2_file, device='cuda'):
        """
        初始化6+k参数化
        6: 3个平移 + 3个旋转
        k: 可旋转键的数量
        """
        self.device = device
        self.mol = Chem.MolFromMol2File(mol2_file, removeHs=False)
        if self.mol is None:
            raise ValueError(f"无法读取mol2文件: {mol2_file}")
        
        # 获取可旋转键
        self.rotatable_bonds = self.get_rotatable_bonds()
        self.k = len(self.rotatable_bonds)
        
        # 初始坐标
        initial_coords_np = self.mol.GetConformer().GetPositions()
        self.initial_coords = torch.tensor(
            initial_coords_np, dtype=torch.float32, device=device, requires_grad=False
        )
        self.num_atoms = len(initial_coords_np)
        
        # 中心点
        center_np = np.mean(initial_coords_np, axis=0)
        self.center = torch.tensor(center_np, dtype=torch.float32, device=device, requires_grad=False)
        
        # 构建分子图信息
        self.bond_graph = self.build_bond_graph()
        
    def get_rotatable_bonds(self):
        """获取可旋转键"""
        rotatable_bonds = []
        for bond in self.mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                if not bond.IsInRing():
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    if a1.GetDegree() > 1 and a2.GetDegree() > 1:
                        rotatable_bonds.append({
                            'bond_idx': bond.GetIdx(),
                            'atom1_idx': a1.GetIdx(),
                            'atom2_idx': a2.GetIdx()
                        })
        return rotatable_bonds
    
    def build_bond_graph(self):
        """构建分子键图"""
        graph = {}
        for bond in self.mol.GetBonds():
            a1, a2 = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            if a1 not in graph:
                graph[a1] = []
            if a2 not in graph:
                graph[a2] = []
            graph[a1].append(a2)
            graph[a2].append(a1)
        return graph
    
    def params_to_coords(self, params):
        """将6+k参数转换为3D坐标"""
        if isinstance(params, np.ndarray):
            params = torch.tensor(params, dtype=torch.float32, device=self.device)
        elif isinstance(params, torch.Tensor) and params.device != self.device:
            params = params.to(self.device)
        
        translation = params[:3]
        rotation = params[3:6]
        torsions = params[6:]
        
        coords = self.initial_coords - translation.unsqueeze(0)
        coords = self.apply_rotation(coords, rotation)
        
        if self.k > 0:
            coords = self.apply_torsions_pytorch(coords, torsions)
        
        return coords
    
    def apply_rotation(self, coords, euler_angles):
        """应用欧拉旋转"""
        alpha, beta, gamma = euler_angles[0], euler_angles[1], euler_angles[2]
        
        cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
        cos_b, sin_b = torch.cos(beta), torch.sin(beta)  
        cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)
        
        Rz = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        Ry = torch.tensor([
            [cos_b, 0, sin_b],
            [0, 1, 0],
            [-sin_b, 0, cos_b]
        ], dtype=torch.float32, device=self.device)
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_g, -sin_g],
            [0, sin_g, cos_g]
        ], dtype=torch.float32, device=self.device)
        
        R = torch.mm(torch.mm(Rz, Ry), Rx)
        
        centered = coords - self.center
        rotated = torch.mm(centered, R.t())
        return rotated + self.center
    
    def apply_torsions_pytorch(self, coords, torsions):
        """应用扭转角"""
        result_coords = coords.clone()
        
        for i, torsion_info in enumerate(self.rotatable_bonds):
            if i >= len(torsions):
                break
                
            torsion_angle = torsions[i]
            atom1_idx = torsion_info['atom1_idx']
            atom2_idx = torsion_info['atom2_idx']
            
            try:
                result_coords = self.apply_single_torsion(
                    result_coords, atom1_idx, atom2_idx, torsion_angle
                )
            except Exception as e:
                print(f"应用扭转角时出错: {e}")
                continue
        
        return result_coords
    
    def apply_single_torsion(self, coords, atom1_idx, atom2_idx, angle):
        """应用单个扭转角"""
        bond_vector = coords[atom2_idx] - coords[atom1_idx]
        bond_axis = bond_vector / torch.norm(bond_vector)
        
        atoms_to_rotate = self.find_atoms_to_rotate(atom2_idx, atom1_idx)
        
        if not atoms_to_rotate:
            return coords
        
        rotation_matrix = self.rodrigues_rotation(bond_axis, angle)
        
        pivot = coords[atom1_idx]
        rotated_coords = coords.clone()
        
        for atom_idx in atoms_to_rotate:
            if atom_idx != atom1_idx and atom_idx != atom2_idx:
                vector_to_pivot = coords[atom_idx] - pivot
                rotated_vector = torch.mv(rotation_matrix, vector_to_pivot)
                rotated_coords[atom_idx] = pivot + rotated_vector
        
        return rotated_coords
    
    def find_atoms_to_rotate(self, start_atom, exclude_atom):
        """找到需要旋转的原子集合"""
        visited = set([exclude_atom])
        to_rotate = set()
        stack = [start_atom]
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                to_rotate.add(current)
                for neighbor in self.bond_graph.get(current, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return to_rotate - set([exclude_atom])
    
    def rodrigues_rotation(self, axis, angle):
        """罗德里格斯旋转公式"""
        axis = axis / torch.norm(axis)
        
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        one_minus_cos = 1 - cos_a
        
        x, y, z = axis[0], axis[1], axis[2]
        
        R = torch.tensor([
            [cos_a + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_a, x*z*one_minus_cos + y*sin_a],
            [y*x*one_minus_cos + z*sin_a, cos_a + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_a],
            [z*x*one_minus_cos - y*sin_a, z*y*one_minus_cos + x*sin_a, cos_a + z*z*one_minus_cos]
        ], dtype=torch.float32, device=self.device)
        
        return R