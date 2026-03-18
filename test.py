import sys
# 在test.py开头添加（根据实际路径调整）
sys.path.append("/storage/zhangjinqiao/UniPoseScore")  # 项目根目录
from core.config import GraphormerConfig  # 替换为实际的类路径

import torch
import os

def rename_checkpoint_keys(original_ckpt_path, new_ckpt_path):
    """
    修改checkpoint中的参数键名，并保存为新文件
    
    Args:
        original_ckpt_path: 原模型权重文件路径（如 xxx.pt/xxx.pth）
        new_ckpt_path: 新模型权重文件保存路径
    """
    # 1. 加载原checkpoint（根据你的环境选择是否指定device）
    # 如果是GPU训练的模型，CPU加载需加 map_location='cpu'
    checkpoint = torch.load(original_ckpt_path, map_location='cpu')
    
    # 2. 获取模型参数的state_dict
    state_dict = checkpoint['model_state_dict']
    
    # 3. 批量替换键名
    new_state_dict = {}
    for old_key, value in state_dict.items():
        # 替换 energy_proj 为 rmsd_proj
        new_key = old_key.replace("energy_proj", "rmsd_proj")
        # 替换 energy_weights 为 rmsd_weights
        new_key = new_key.replace("energy_weights", "rmsd_weights")
        new_state_dict[new_key] = value
    
    # 4. 更新checkpoint中的state_dict
    checkpoint['model_state_dict'] = new_state_dict
    
    # 5. 保存新的checkpoint
    torch.save(checkpoint, new_ckpt_path)
    print(f"✅ 新模型文件已保存至：{new_ckpt_path}")
    print(f"📝 共修改 {len(new_state_dict)} 个参数键名")

# ==================== 调用示例 ====================
if __name__ == "__main__":
    # 请替换为你的实际文件路径
    ORIGINAL_CKPT = "/storage/zhangjinqiao/UniPoseScore/core/model/checkpoint_stage1_epoch140_best.pt"  # 原权重文件
    NEW_CKPT = "/storage/zhangjinqiao/UniPoseScore/core/model/model.pt"            # 新权重文件保存路径
    
    # 检查原文件是否存在
    if not os.path.exists(ORIGINAL_CKPT):
        print(f"❌ 原文件不存在：{ORIGINAL_CKPT}")
    else:
        rename_checkpoint_keys(ORIGINAL_CKPT, NEW_CKPT)