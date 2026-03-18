import shutil
import os
import torch
import numpy as np
import re
from rdkit import Chem
import torch.optim as optim
from .ligand_6k_parameterizer import Ligand6KParameterizer
from .structure_optimizer import StructureOptimizer
from collections import OrderedDict, defaultdict
from core.feature_extractor import extract_complex_features
from core.utils import save_optimized_structure
import time
import sys
from io import StringIO
import tempfile

class IterativeGraphormerOptimizer:
    """迭代优化器"""
    
    def __init__(self, model_predictor, device='cuda'):
        self.predictor = model_predictor
        self.device = device
    
    def predict_for_structure(self, ligand_mol2, receptor_pdb=None, sample_id=None):
        """对单个结构进行预测"""
        try:
            features = extract_complex_features(receptor_pdb, ligand_mol2)
            
            n_atoms = features['atoms'].size(0)
            
            # 确保所有特征都有批次维度
            atoms = features['atoms'].unsqueeze(0).to(self.device) 
            tags = features['tags'].unsqueeze(0).to(self.device) 
            pos = features['pos'].unsqueeze(0).to(self.device) 
            real_mask = torch.ones(1, n_atoms, dtype=torch.bool, device=self.device)
            
            with torch.no_grad():
                energy_pred, displacements_pred = self.predictor.model(
                    atoms=atoms,
                    tags=tags,
                    pos=pos,
                    real_mask=real_mask
                )
            
            # 提取配体位移
            n_ligand_atoms = torch.sum(features['tags'] == 1).item()
            if displacements_pred.shape[1] != n_ligand_atoms:
                if displacements_pred.shape[1] > n_ligand_atoms:
                    displacements_pred = displacements_pred[:, :n_ligand_atoms, :]
                else:
                    padding = torch.zeros(
                        displacements_pred.shape[0], 
                        n_ligand_atoms - displacements_pred.shape[1],
                        displacements_pred.shape[2],
                        device=displacements_pred.device
                    )
                    displacements_pred = torch.cat([displacements_pred, padding], dim=1)
            
            return energy_pred.item(), displacements_pred.squeeze(0).cpu().numpy(), features
            
        except Exception as e:
            print(f"预测失败: {e}")
            raise
    
    def optimize_single_iteration(self, input_mol2, output_mol2, displacements_pred, max_iterations=100):
        """单轮6+k优化"""
        try:
            
            parameterizer = Ligand6KParameterizer(input_mol2, device=self.device)
            
            if isinstance(displacements_pred, torch.Tensor):
                displacements_pred = displacements_pred.detach().cpu().numpy()
            
            # 确保位移形状匹配
            if displacements_pred.shape[0] != parameterizer.num_atoms:
                if displacements_pred.shape[0] > parameterizer.num_atoms:
                    displacements_pred = displacements_pred[:parameterizer.num_atoms]
                else:
                    padding = np.zeros((parameterizer.num_atoms - displacements_pred.shape[0], 3))
                    displacements_pred = np.vstack([displacements_pred, padding])
            
            # 执行优化
            optimizer = StructureOptimizer(parameterizer, displacements_pred, None, device=self.device)
            optimized_params, final_loss, success = optimizer.optimize_with_restarts()
            
            if success:
                optimized_coords = parameterizer.params_to_coords(
                    torch.tensor(optimized_params, device=self.device)
                ).detach().cpu().numpy()
                
                save_success = save_optimized_structure(input_mol2, optimized_coords, output_mol2)
                
                if save_success:
                    return True, final_loss, 0.0
                else:
                    return False, final_loss, 0.0
            else:
                import shutil
                shutil.copy2(input_mol2, output_mol2)
                return False, final_loss, 0.0
                
        except Exception as e:
            print(f"单轮优化失败: {e}")
            import shutil
            shutil.copy2(input_mol2, output_mol2)
            return False, float('inf'), 0.0
    
    def simple_iterative_optimization(self, initial_mol2, output_dir, receptor_pdb=None,
                                  max_cycles=10, max_iterations_per_cycle=100,
                                  early_stop_threshold=0.001):
    
        import os
        import shutil
        import tempfile
        
        # 检查输入文件是否存在
        if not os.path.exists(initial_mol2):
            print(f"错误: 输入配体文件不存在 - {initial_mol2}")
            return None, 0.0, 0.0, 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始优化: {os.path.basename(initial_mol2)}")
        if receptor_pdb:
            print(f"使用受体: {os.path.basename(receptor_pdb)}")
        print(f"最大循环数: {max_cycles}")
        print("-" * 60)
        
        # 初始化变量
        current_pred_rmsd = 0.0
        best_pred_rmsd = float('inf')
        best_structure_path = None
        n_cycles = 0
        
        with tempfile.TemporaryDirectory(prefix="opt_cycle_", dir=output_dir) as temp_dir:
            # 处理当前输入和输出的文件
            current_input = os.path.join(temp_dir, "current_input.mol2")
            current_output = os.path.join(temp_dir, "current_output.mol2")
            
            # 复制初始结构
            try:
                shutil.copy2(initial_mol2, current_input)
            except Exception as e:
                print(f"复制初始结构失败: {e}")
                return None, 0.0, 0.0, 0
            
            # 计算初始结构RMSD
            print("计算初始预测RMSD...")
            try:
                initial_pred_rmsd, initial_displacements, _ = self.predict_for_structure(
                    current_input, receptor_pdb, "initial"
                )
                print(f"初始预测RMSD: {initial_pred_rmsd:.3f}Å")
            except Exception as e:
                print(f"计算初始预测RMSD失败: {e}")
                return None, 0.0, 0.0, 0
            
            # 初始化
            best_pred_rmsd = initial_pred_rmsd
            best_structure_path = current_input
            current_pred_rmsd = initial_pred_rmsd
            current_displacements = initial_displacements
            
            previous_pred_rmsd = current_pred_rmsd
            
            for cycle in range(max_cycles):
                print(f"\n循环 {cycle+1}/{max_cycles}")
                
                # 优化当前结构
                print("  执行优化...")
                success, final_loss, opt_distance = self.optimize_single_iteration(
                    input_mol2=current_input,
                    output_mol2=current_output,
                    displacements_pred=current_displacements,
                    max_iterations=max_iterations_per_cycle
                )
                
                if not success or not os.path.exists(current_output):
                    print(f"  优化失败或输出文件未创建")
                    break
                
                # 预测优化后结构的RMSD
                print("  预测优化后结构...")
                try:
                    optimized_pred, optimized_displacements, _ = self.predict_for_structure(
                        current_output, receptor_pdb, f"cycle_{cycle}"
                    )
                    print(f"  优化后预测RMSD: {optimized_pred:.3f}Å")
                except Exception as e:
                    print(f"  预测优化后结构失败: {e}")
                    break
                
                # 更新最佳结构
                if optimized_pred < best_pred_rmsd:
                    best_pred_rmsd = optimized_pred
                    # 保存最佳结构副本
                    best_structure_path = os.path.join(temp_dir, f"best_cycle_{cycle}.mol2")
                    shutil.copy2(current_output, best_structure_path)
                    print(f"  新的最佳RMSD: {best_pred_rmsd:.3f}Å")
                
                # 准备下一次循环
                # 交换文件：当前输出变为下一轮输入
                temp_input = os.path.join(temp_dir, f"temp_input_{cycle}.mol2")
                shutil.move(current_output, temp_input)
                shutil.move(temp_input, current_input)
                
                current_pred_rmsd = optimized_pred
                current_displacements = optimized_displacements
                n_cycles = cycle + 1
                
                # 收敛检查
                if cycle > 0:
                    rmsd_change = abs(current_pred_rmsd - previous_pred_rmsd)
                    if rmsd_change < early_stop_threshold:
                        print(f"  收敛条件满足, RMSD变化: {rmsd_change:.3f}Å < {early_stop_threshold}Å")
                        break
                
                previous_pred_rmsd = current_pred_rmsd
            
            # 保存最终结果
            final_output = os.path.join(output_dir, "optimized_final.mol2")
            
            if best_structure_path and os.path.exists(best_structure_path):
                try:
                    shutil.copy2(best_structure_path, final_output)
                except Exception as e:
                    print(f"保存最终结构失败: {e}")
                    # 回退到初始结构
                    shutil.copy2(initial_mol2, final_output)
            else:
                # 没有优化结果，使用初始结构
                shutil.copy2(initial_mol2, final_output)
        
        # 如果需要验证最终结果，可以在这里计算一次
        print(f"\n优化完成:")
        print(f"  总循环数: {n_cycles}")
        print(f"  初始预测RMSD: {initial_pred_rmsd:.3f}Å")
        print(f"  最佳预测RMSD: {best_pred_rmsd:.3f}Å")
        print(f"  RMSD改进: {initial_pred_rmsd - best_pred_rmsd:+.3f}Å")
    
        return final_output, initial_pred_rmsd, best_pred_rmsd, n_cycles