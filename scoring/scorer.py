import numpy as np
import pandas as pd
import os
from typing import List, Tuple

def batch_scoring(predictor, data_loader, save_results, output_dir):
    """
    批量评分函数
    
    参数:
    - predictor: GraphormerPredictor实例
    - data_loader: 数据加载器
    - save_results: 是否保存结果
    - output_dir: 输出目录
    
    返回:
    - sample_ids: 样本ID列表
    - pred_scores: 预测RMSD列表
    - pred_displacements: 预测位移列表
    """
    # 执行预测
    sample_ids, pred_scores, pred_displacements = predictor.predict(data_loader)
    
    # 保存结果
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存CSV结果
        df_results = pd.DataFrame({
            'sample_id': sample_ids,
            'predicted_rmsd': pred_scores
        })
        
        csv_path = os.path.join(output_dir, "scoring_results.csv")
        df_results.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"评分结果已保存至: {csv_path}")
        
    
    return sample_ids, pred_scores, pred_displacements