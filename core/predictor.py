import torch
import numpy as np
from .model import Graphormer3D

class GraphormerPredictor:
    def __init__(self, model_path, config, device='cuda'):
        self.device = device
        self.config = config
        self.model = Graphormer3D(config).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, data_loader):
        all_preds = []
        all_ids = []
        all_pred_displacements = []

        
        with torch.no_grad():
            for batch in data_loader:
                atoms = batch['atoms'].to(self.device)
                tags = batch['tags'].to(self.device)
                pos = batch['pos'].to(self.device)
                real_mask = ~batch['padding_mask'].to(self.device)

                rmsd_pred, displacements_pred = self.model(
                    atoms=atoms,
                    tags=tags,
                    pos=pos,
                    real_mask=real_mask
                )
                
                preds = rmsd_pred.squeeze().cpu().numpy()
                if preds.ndim == 0:
                    preds = np.expand_dims(preds, axis=0)
                all_preds.extend(preds)

                all_ids.extend(batch['sample_pose_ids'])
                
                batch_size = pos.shape[0]
                for i in range(batch_size):
                    n_node = tags[i].sum().item()
                    if n_node > 0:
                        all_pred_displacements.append(displacements_pred[i, :n_node].cpu().numpy())

        return all_ids, np.array(all_preds), np.array(all_pred_displacements)