import torch
import numpy as np
from scipy.optimize import minimize
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.optim as optim

class StructureOptimizer:
    """基于6+k参数的PyTorch结构优化器"""
    
    def __init__(self, parameterizer, predicted_displacements, receptor_coords=None, device='cuda'):
        self.parameterizer = parameterizer
        self.device = device
        if isinstance(predicted_displacements, np.ndarray):
            self.predicted_displacements = torch.tensor(
                predicted_displacements, dtype=torch.float32, device=device, requires_grad=False
            )
        else:
            self.predicted_displacements = predicted_displacements.to(device)
        
        if receptor_coords is not None:
            if isinstance(receptor_coords, np.ndarray):
                self.receptor_coords = torch.tensor(
                    receptor_coords, dtype=torch.float32, device=device, requires_grad=False
                )
            elif isinstance(receptor_coords, torch.Tensor) and receptor_coords.device != device:
                self.receptor_coords = receptor_coords.to(device)
            else:
                self.receptor_coords = receptor_coords
        else:
            self.receptor_coords = None
        
        self.weights = {
            'displacement': 1.0,
            'intra_clash': 0.3,
            'inter_clash': 0.5,
            'smoothness': 0.1
        }
    
    def objective_function(self, params):
        """PyTorch目标函数"""
        if isinstance(params, np.ndarray):
            params = torch.tensor(params, dtype=torch.float32, device=self.device, requires_grad=True)
        elif isinstance(params, torch.Tensor) and params.device != self.device:
            params = params.to(self.device)
        
        current_coords = self.parameterizer.params_to_coords(params)
        
        displacement_loss = self.calculate_displacement_loss(current_coords)
        intra_clash_loss = self.calculate_intra_clash_loss(current_coords)
        inter_clash_loss = self.calculate_inter_clash_loss(current_coords)
        smoothness_loss = self.calculate_smoothness_loss(params)
        
        total_loss = (self.weights['displacement'] * displacement_loss +
                     self.weights['intra_clash'] * intra_clash_loss +
                     self.weights['inter_clash'] * inter_clash_loss +
                     self.weights['smoothness'] * smoothness_loss)
        
        return total_loss
    
    def calculate_displacement_loss(self, current_coords):
        """计算与预测位移的匹配度"""
        target_coords = self.parameterizer.initial_coords - self.predicted_displacements
        diff = current_coords - target_coords
        squared_distances = torch.sum(diff**2, dim=1)
        rmsd = torch.sqrt(torch.mean(squared_distances))
        return rmsd
    
    def calculate_intra_clash_loss(self, coords, clash_threshold=1.8):
        """计算分子内原子碰撞"""
        n_atoms = coords.shape[0]
        if n_atoms <= 1:
            return torch.tensor(0.0, device=self.device)
        
        distances = torch.cdist(coords, coords)
        mask = torch.triu(torch.ones(n_atoms, n_atoms, device=self.device), diagonal=1).bool()
        relevant_distances = distances[mask]
        
        clash_mask = relevant_distances < clash_threshold
        if not torch.any(clash_mask):
            return torch.tensor(0.0, device=self.device)
        
        clash_penalties = 1.0 / (relevant_distances[clash_mask] + 1e-6) - 1.0 / clash_threshold
        clash_penalties = torch.clamp(clash_penalties, min=0)
        
        return torch.mean(clash_penalties) * 0.1
    
    def calculate_inter_clash_loss(self, ligand_coords, clash_threshold=2.0):
        """计算配体-受体碰撞"""
        if self.receptor_coords is None:
            return torch.tensor(0.0, device=self.device)
        
        receptor_sample = self.receptor_coords[:min(100, len(self.receptor_coords))]
        distances = torch.cdist(ligand_coords, receptor_sample)
        
        clash_mask = distances < clash_threshold
        if not torch.any(clash_mask):
            return torch.tensor(0.0, device=self.device)
        
        clash_penalties = 1.0 / (distances[clash_mask] + 1e-6) - 1.0 / clash_threshold
        clash_penalties = torch.clamp(clash_penalties, min=0)
        
        return torch.mean(clash_penalties) * 0.1
    
    def calculate_smoothness_loss(self, params):
        """惩罚参数的大幅度变化"""
        if isinstance(params, torch.Tensor):
            return torch.sum(params**2) * 0.001
        else:
            return np.sum(params**2) * 0.001
    
    def displacement_based_initialization(self, predicted_displacements, initial_coords):
        """基于位移分析的精确参数初始化"""
        if hasattr(predicted_displacements, 'cpu'):
            predicted_displacements_np = predicted_displacements.cpu().numpy()
        else:
            predicted_displacements_np = np.array(predicted_displacements)
        
        if hasattr(initial_coords, 'cpu'):
            initial_coords_np = initial_coords.cpu().numpy()
        else:
            initial_coords_np = np.array(initial_coords)
        
        from scipy.spatial.transform import Rotation as R
        
        centroid_displacement = np.mean(predicted_displacements_np, axis=0)
        
        try:
            target_coords = initial_coords_np - predicted_displacements_np
            centered_initial = initial_coords_np - np.mean(initial_coords_np, axis=0)
            centered_target = target_coords - np.mean(target_coords, axis=0)
            
            H = centered_initial.T @ centered_target
            U, S, Vt = np.linalg.svd(H)
            rotation_matrix = Vt.T @ U.T
            
            if np.linalg.det(rotation_matrix) < 0:
                Vt[2, :] *= -1
                rotation_matrix = Vt.T @ U.T
            
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz')
            
        except Exception as e:
            print(f"旋转分析失败: {e}")
            euler_angles = np.zeros(3)
        
        params = np.zeros(6 + self.parameterizer.k)
        params[0:3] = centroid_displacement * 0.9
        params[3:6] = euler_angles * 0.1
        
        if self.parameterizer.k > 0:
            params[6:] = np.random.normal(0, 0.1, self.parameterizer.k)
        
        return torch.tensor(params, dtype=torch.float32, device=self.device, requires_grad=True)
    
    def smart_parameter_initialization(self, predicted_displacements):
        """基于预测位移智能初始化6+k参数"""
        if hasattr(predicted_displacements, 'cpu'):
            predicted_displacements = predicted_displacements.cpu().numpy()
        else:
            predicted_displacements = np.array(predicted_displacements)
        
        mean_displacement = np.mean(predicted_displacements, axis=0)
        displacement_norms = np.linalg.norm(predicted_displacements, axis=1)
        max_displacement_idx = np.argmax(displacement_norms)
        principal_direction = predicted_displacements[max_displacement_idx]
        principal_direction = principal_direction / (np.linalg.norm(principal_direction) + 1e-6)
        
        params = np.zeros(6 + self.parameterizer.k)
        params[0:3] = mean_displacement * 0.5
        
        if np.linalg.norm(principal_direction) > 0.1:
            params[3] = np.arctan2(principal_direction[1], principal_direction[0]) * 0.1
            params[4] = np.arcsin(principal_direction[2]) * 0.1
        else:
            params[3:6] = np.random.normal(0, 0.1, 3)
        
        if self.parameterizer.k > 0:
            params[6:] = np.random.normal(0, 0.1, self.parameterizer.k)
        
        return torch.tensor(params, dtype=torch.float32, device=self.device, requires_grad=True)
    
    def fallback_optimization(self):
        """备用优化策略"""
        print("使用备用优化策略: 梯度下降")
        
        try:
            import torch
            import torch.optim as optim
            
            params_tensor = torch.zeros(6 + self.parameterizer.k, device=self.device, requires_grad=True)
            optimizer = optim.Adam([params_tensor], lr=0.01)
            
            best_loss = None
            best_params = None
            patience = 10
            no_improvement = 0
            
            warmup_losses = []
            for warmup_epoch in range(5):
                optimizer.zero_grad()
                current_params = params_tensor.detach().numpy().copy()
                loss_value = self.objective_function(current_params)
                loss_tensor = torch.tensor(loss_value, requires_grad=True, dtype=torch.float32)
                loss_tensor.backward()
                optimizer.step()
                warmup_losses.append(loss_value)
            
            if warmup_losses:
                best_loss = min(warmup_losses)
                best_params = params_tensor.detach().numpy().copy()
            
            for epoch in range(100):
                optimizer.zero_grad()
                current_params = params_tensor.detach().numpy().copy()
                loss_value = self.objective_function(current_params)
                loss_tensor = torch.tensor(loss_value, requires_grad=True, dtype=torch.float32)
                loss_tensor.backward()
                optimizer.step()
                
                if best_loss is None or loss_value < best_loss - 1e-4:
                    best_loss = loss_value
                    best_params = params_tensor.detach().numpy().copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                if no_improvement >= patience:
                    break
            
            if best_params is not None:
                final_loss = self.objective_function(best_params)
                initial_loss = self.objective_function(np.zeros_like(best_params))
                
                if final_loss < initial_loss - 1e-4:
                    return best_params, final_loss, True
                else:
                    return np.zeros(6 + self.parameterizer.k), initial_loss, False
            else:
                return np.zeros(6 + self.parameterizer.k), 10.0, False
                
        except Exception as e:
            print(f"备用优化过程中出错: {e}")
            return np.zeros(6 + self.parameterizer.k), 10.0, False

    def optimize(self, max_iter=200, lr=0.1, patience=10):
        """执行PyTorch优化"""
        print(f"开始PyTorch优化，参数维度: {6 + self.parameterizer.k}")
        
        params = self.displacement_based_initialization(self.predicted_displacements, self.parameterizer.initial_coords)
        optimizer = optim.Adam([params], lr=lr)
        
        best_loss = float('inf')
        best_params = None
        no_improvement = 0
        loss_history = []
        
        for epoch in range(max_iter):
            optimizer.zero_grad()
            loss = self.objective_function(params)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                best_params = params.detach().clone()
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= patience:
                break
        
        if best_params is not None:
            final_loss = self.objective_function(best_params).item()
            return best_params.cpu().numpy(), final_loss, True
        else:
            return np.zeros(6 + self.parameterizer.k), float('inf'), False
        
    def optimize_with_restarts(self, n_restarts=2, max_iter=100, initial_lr=0.5):
        """带随机重启的优化"""
        best_overall_loss = float('inf')
        best_overall_params = None
        
        for restart in range(n_restarts):
            if restart == 0:
                params = self.displacement_based_initialization(
                    self.predicted_displacements, 
                    self.parameterizer.initial_coords
                )
            else:
                params = self.smart_parameter_initialization(self.predicted_displacements)
            
            params.requires_grad_(True)
            optimizer = optim.Adam([params], lr=initial_lr)
            
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_iter//2, eta_min=initial_lr*0.1
            )
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, verbose=False
            )
            
            best_loss = float('inf')
            best_params = None
            
            for epoch in range(max_iter):
                optimizer.zero_grad()
                loss = self.objective_function(params)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                
                if epoch < max_iter // 2:
                    cosine_scheduler.step()
                else:
                    plateau_scheduler.step(current_loss)
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = params.detach().clone()
            
            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_params = best_params
        
        if best_overall_params is not None:
            return best_overall_params.cpu().numpy(), best_overall_loss, True
        else:
            return np.zeros(6 + self.parameterizer.k), float('inf'), False