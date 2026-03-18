#!/usr/bin/env python3
"""
 蛋白质-配体复合物评分与构象优化
提供 `score` 和 `optimize` 两种模式。
"""

import argparse
import os
import sys
import torch
import time
import numpy as np
from torch.utils.data import DataLoader

from core.config import GraphormerConfig
from core.predictor import GraphormerPredictor
from core.utils import collate_fn, calculate_dockrmsd

from scoring.dataset import Graphormer3DDataset
from scoring.scorer import batch_scoring
from optimization.iterative_optimizer import IterativeGraphormerOptimizer


def setup_score_parser(subparsers):
    """配置 `score` 子命令的解析器"""
    parser_score = subparsers.add_parser('score', help='仅对输入的复合物进行RMSD预测')
    parser_score.add_argument("--model_path", type=str, required=True,
                             help='训练好的模型路径 (.pt)')
    parser_score.add_argument("--protein_pdb", type=str, required=True,
                             help='蛋白质受体文件路径 (.pdb)')
    parser_score.add_argument("--ligand_mol2", type=str, required=True,
                             help='配体构象文件路径 (.mol2)')
    parser_score.add_argument("--batch_size", type=int, default=4,
                             help='预测批大小')
    parser_score.add_argument("--output_dir", type=str, default="./test_output",
                             help='结果输出目录')
    parser_score.add_argument("--use_cuda", action='store_true', default=True,
                             help='使用GPU进行计算')
    return parser_score


def setup_optimize_parser(subparsers):
    """配置 `optimize` 子命令的解析器 - 简化无参考版本"""
    parser_opt = subparsers.add_parser('optimize', help='对输入构象进行迭代优化')
    parser_opt.add_argument("--model_path", type=str, required=True,
                           help='训练好的模型路径 (.pt)')
    parser_opt.add_argument("--protein_pdb", type=str, required=True,
                           help='蛋白质受体文件路径 (.pdb)')
    parser_opt.add_argument("--ligand_mol2", type=str, required=True,
                           help='待优化的配体初始构象文件路径 (.mol2)')
    parser_opt.add_argument("--output_dir", type=str, default="./test_output",
                           help='优化结果输出目录')
    parser_opt.add_argument("--max_cycles", type=int, default=10,
                           help='最大优化循环次数')
    parser_opt.add_argument("--max_iterations_per_cycle", type=int, default=100,
                           help='每个优化循环内的最大迭代次数')
    parser_opt.add_argument("--early_stop_threshold", type=float, default=0.001,
                           help='RMSD变化收敛阈值 (Å)')
    parser_opt.add_argument("--use_cuda", action='store_true', default=True,
                           help='使用GPU进行计算')
    return parser_opt

def run_score_mode(args):
    """执行评分模式"""
    print(f"{'='*60}")
    print("评分模式")
    print(f"{'='*60}")

    # 1. 参数检查和路径准备
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")

    if not os.path.exists(args.ligand_mol2):
        print(f"错误: 配体文件不存在 - {args.ligand_mol2}")
        sys.exit(1)
    if args.protein_pdb and not os.path.exists(args.protein_pdb):
        print(f"警告: 受体文件不存在，将以配体单独模式运行 - {args.protein_pdb}")

    # 2. 初始化模型
    print("\n[1/3] 正在加载模型...")
    config = GraphormerConfig()
    predictor = GraphormerPredictor(args.model_path, config, device)

    # 3. 检查是否为多构象MOL2文件
    is_multi_mol2 = False
    with open(args.ligand_mol2, 'r') as f:
        content = f.read()
        mol_count = content.count('@<TRIPOS>MOLECULE')
        if mol_count > 1:
            is_multi_mol2 = True
            print(f"检测到多构象MOL2文件，包含 {mol_count} 个构象")

    # 4. 准备数据
    print("\n[2/3] 正在处理输入结构...")
    
    if is_multi_mol2:
        # 使用多构象模式
        dataset = Graphormer3DDataset.from_multi_mol2_file(
            protein_pdb=args.protein_pdb,
            multi_mol2_path=args.ligand_mol2,
            temp_dir=os.path.join(args.output_dir, "temp_split")
        )
    else:
        # 单分子模式
        dataset = Graphormer3DDataset.from_raw_files(
            protein_pdb_list=[args.protein_pdb],
            ligand_mol2_list=[args.ligand_mol2]
        )
        
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 5. 执行预测
    print("\n[3/3] 正在进行预测...")
    
    # batch_scoring 返回 (ids, preds, pred_disps)
    sample_ids, pred_scores, pred_displacements = batch_scoring(
        predictor, 
        data_loader,
        save_results=True,
        output_dir=args.output_dir
    )


    # 6. 输出结果
    result_file = os.path.join(args.output_dir, "scoring_results.csv")
    
    with open(result_file, 'w') as f:
        f.write("pose_id, predicted_rmsd\n")
        
        for i, (sid, score) in enumerate(zip(sample_ids, pred_scores)):
            # 解析sample_pose_ids
            if ' ' in sid:
                sample_id, pose_id = sid.split(' ', 1)
            else:
                sample_id = sid
                pose_id = f"pose_{i+1:04d}"
            
            f.write(f"{pose_id},{score:.4f}\n")

    print(f"\n{'='*60}")
    print("评分完成!")
    print(f"输入文件: {os.path.basename(args.ligand_mol2)}")
    
    if is_multi_mol2:
        print(f"处理模式: 多构象 ({len(sample_ids)} 个构象)")
    else:
        print(f"处理模式: 单构象")
    
    if args.protein_pdb:
        print(f"输入受体: {os.path.basename(args.protein_pdb)}")
    
    print(f"详细结果已保存至: {result_file}")
    
    # 清理临时文件
    if is_multi_mol2 and hasattr(dataset, 'cleanup_temp_files'):
        dataset.cleanup_temp_files()
    
    print(f"{'='*60}")



def run_optimize_mode(args):
    """执行优化模式 - 简化无参考版本"""
    print(f"{'='*60}")
    print("优化模式")
    print(f"{'='*60}")
    
    # 检查必要文件
    if not os.path.exists(args.ligand_mol2):
        print(f"错误: 配体文件不存在 - {args.ligand_mol2}")
        sys.exit(1)
    
    # 检查受体文件（可选）
    has_receptor = args.protein_pdb and os.path.exists(args.protein_pdb)
    if args.protein_pdb and not has_receptor:
        print(f"警告: 受体文件不存在，优化将不考虑口袋 - {args.protein_pdb}")
        has_receptor = False
    else:
        print(f"使用受体: {os.path.basename(args.protein_pdb)}" if has_receptor else "运行模式: 无受体优化")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型和优化器
    print("\n[1/3] 正在加载模型...")
    config = GraphormerConfig()
    predictor = GraphormerPredictor(args.model_path, config, device)
    iterative_optimizer = IterativeGraphormerOptimizer(predictor, device)
    
    # 执行迭代优化
    print(f"\n[2/3] 开始迭代优化 (最大 {args.max_cycles} 轮)...")
    sample_id = os.path.basename(args.ligand_mol2).replace('.mol2', '')
    
    final_structure, initial_pred_rmsd, final_pred_rmsd, n_cycles = iterative_optimizer.simple_iterative_optimization(
        output_dir=args.output_dir,
        initial_mol2=args.ligand_mol2,
        receptor_pdb=args.protein_pdb if has_receptor else None,
        max_cycles=args.max_cycles,
        max_iterations_per_cycle=args.max_iterations_per_cycle,
        early_stop_threshold=args.early_stop_threshold
    )
    
    # 输出结果
    print(f"\n[3/3] 优化完成!")
    print(f"{'='*60}")
    print("优化结果摘要:")
    print(f"  输入结构: {os.path.basename(args.ligand_mol2)}")
    print(f"  初始预测RMSD: {initial_pred_rmsd:.3f}Å")
    print(f"  最终预测RMSD: {final_pred_rmsd:.3f}Å")
    print(f"  优化循环数: {n_cycles}")
    print(f"{'='*60}")
    
    # 保存结果文件
    result_file = os.path.join(args.output_dir, "optimization_summary.txt")
    with open(result_file, 'w') as f:
        f.write("优化结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"输入结构: {os.path.basename(args.ligand_mol2)}\n")
        f.write(f"初始预测RMSD: {initial_pred_rmsd:.3f} Å\n")
        f.write(f"最终预测RMSD: {final_pred_rmsd:.3f} Å\n")
        f.write(f"输出结构: {os.path.basename(final_structure)}\n")
    
    print(f"结果摘要已保存至: {result_file}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 仅评分模式
  python main.py score \\
    --model_path /path/to/model.pth \\
    --protein_pdb /path/to/protein.pdb \\
    --ligand_mol2 /path/to/ligand.mol2 \\
    --output_dir ./test_output

  # 2. 构象优化模式
  python main.py optimize \\
    --model_path /path/to/model.pth \\
    --protein_pdb /path/to/protein.pdb \\
    --ligand_mol2 /path/to/initial_pose.mol2 \\
    --output_dir ./test_output \\
        """
    )

    subparsers = parser.add_subparsers(dest='command', title='可用命令', required=True)
    setup_score_parser(subparsers)
    setup_optimize_parser(subparsers)

    args = parser.parse_args()

    if args.command == 'score':
        run_score_mode(args)
    elif args.command == 'optimize':
        run_optimize_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
