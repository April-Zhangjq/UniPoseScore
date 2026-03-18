# UniPoseScore
UniPoseScore: A Unified Graphormer3D-based Framework for Protein–Ligand Binding Pose Scoring and Refinement


## Installation
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Scoring
Score a protein-ligand complex:

```bash
python main.py score \
  --model_path ./core/model/model.pt \
  --protein_pdb ./examples/1h22/1h22_protein.pdb \
  --ligand_mol2 ./examples/1h22/1h22_ligand.mol2 \
  --output_dir ./test_output/1h22
```

Score multiple ligands:

```bash
python main.py score \
  --model_path ./core/model/model.pt \
  --protein_pdb ./examples/1h22/1h22_protein.pdb \
  --ligand_mol2 ./examples/1h22/1h22_decoys.mol2 \
  --output_dir ./test_output/1h22
```

### 2. Optimization
Optimize ligand conformation:

```bash
python main.py optimize \
  --model_path ./core/model/model.pt \
  --protein_pdb ./examples/1h22/1h22_protein.pdb \
  --ligand_mol2 ./examples/1h22/1h22_1.mol2 \
  --output_dir ./test_output/1h22_1 \
  --max_cycles 10 \
  --early_stop_threshold 0.001
```

## Required Arguments
- `--model_path`: Path to trained model (.pt)
- `--protein_pdb`: Protein structure file (.pdb)
- `--ligand_mol2`: Ligand structure file (.mol2) - single or multiple
- `--output_dir`: Output directory

## Optional Arguments (optimize only)
- `--max_cycles`: Maximum optimization cycles (default: 10)
- `--early_stop_threshold`: Early Stop threshold (default: 0.001)

## Input Format
- Protein: PDB format
- Ligand: MOL2 format
  - For **scoring**: Can contain single or multiple poses
  - For **optimization**: Must contain only a single pose
  
## Output
- `score_results.csv`: Scoring results
- `optimized_structures/`: Optimized conformations

## Example Data
See `./examples/1h22/` for sample files.



