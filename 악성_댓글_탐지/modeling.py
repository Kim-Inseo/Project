import torch
import json
from config import ConfigModel
import torch.nn.functional as F
from models import CustomModel
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __getitem__(self, idx):
        item = {}
        item['embeddings'] = torch.Tensor(self.embeddings[idx])
        return item

    def __len__(self):
        return len(self.embeddings)


def classify(text_tokens_list):
    batch_size = 1
    dataset = CustomDataset(text_tokens_list)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    config_model = ConfigModel(model_path='./models/checkpoint.pt',
                               var_models_path='./models/var_models.json')

    model_path = config_model.model_path
    var_models_path = config_model.var_models_path

    with open(var_models_path, 'r') as f:
        var_models_dict = json.load(f)

    device = var_models_dict['device']
    model = CustomModel(embed_dim=var_models_dict['embed_dim'],
                        hidden_dim=var_models_dict['hidden_dim'],
                        output_dim=var_models_dict['output_dim'],
                        device=device,
                        num_layers=var_models_dict['num_layers'],
                        bidirectional=var_models_dict['bidirectional']).to(device)

    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result_list = []
        for batch in data_loader:
            inputs = batch['embeddings'].to(device)
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=-1)
            max_val_list, predict_list = torch.max(prob.data, -1)
            is_malicious = ['악성 댓글' if predict else '악성 댓글 아님' for predict in predict_list]

            result = list(zip(is_malicious, max_val_list))
            result_list += result

    return result_list

