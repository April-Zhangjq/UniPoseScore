# only score
python main.py score --model_path ./core/model/model.pt --protein_pdb ./examples/1h22/1h22_protein.pdb --ligand_mol2 ./examples/1h22/1h22_ligand.mol2 --output_dir ./test_output/1h22
python main.py score --model_path ./core/model/model.pt --protein_pdb ./examples/1h22/1h22_protein.pdb --ligand_mol2 ./examples/1h22/1h22_decoys.mol2 --output_dir ./test_output/1h22

# optimize
python main.py optimize --model_path ./core/model/model.pt --protein_pdb ./examples/1h22/1h22_protein.pdb --ligand_mol2 ./examples/1h22/1h22_1.mol2 --output_dir ./test_output/1h22_1 --max_cycles 10 --early_stop_threshold 0.001
