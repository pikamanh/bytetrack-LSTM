import torch.nn.functional as F
import torch.nn as nn

def get_kan_model(model_name, layer_dims):
    model_name = model_name.lower()
    if model_name == 'faster_kan':
        from models.faster_kan import FasterKAN
        return FasterKAN(layer_dims)
    elif model_name == 'fast_kan':
        from models.fast_kan import FastKAN
        return FastKAN(layer_dims)
    elif model_name == 'bsrbf_kan':
        from models.bsrbf_kan import BSRBF_KAN
        return BSRBF_KAN(layer_dims)
    elif model_name == 'efficient_kan':
        from models.efficient_kan import KAN
        return KAN(layer_dims)
    else:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ!")


class TrajectorySiameseKAN(nn.Module):
    def __init__(self, model_name='faster_kan', input_dim=70, hidden_dim=32, output_dim=16):
        super(TrajectorySiameseKAN, self).__init__()
        self.kan = get_kan_model(model_name, [input_dim, hidden_dim, output_dim])

    def forward(self, x1, x2):
        feat1 = self.kan(x1)
        feat2 = self.kan(x2)
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        return feat1, feat2
    
    def extract_feature(self, x):
        feat = self.kan(x)
        return F.normalize(feat, p=2, dim=1)