import torch
import json
from config import ConfigModel
import torch.nn.functional as F
from models import CustomModel
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    '''
    3차원 list를 받아서
    shape: (문장 개수, sequence 길이, 임베딩 벡터 크기)
    각 batch마다 3차원 Tensor를 반환할 수 있도록 한다
    shape: (batch 크기, sequence 길이, 임베딩 벡터 크기)
    '''
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __getitem__(self, idx):
        item = {}
        item['embeddings'] = torch.Tensor(self.embeddings[idx])
        return item

    def __len__(self):
        return len(self.embeddings)


def classify(text_tokens_list):
    '''
    :param text_tokens_list: 3차원 리스트
                             (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰,
                             세 번째 차원에 단어 토큰에 대응하는 FastText 임베딩 벡터)
                             shape: (문장 개수, sequence 길이, 임베딩 벡터 크기)
    :return: 2차원 리스트
             (첫 번째 차원에 문장, 두 번째 차웜에 각 문장별 악성 댓글 여부와 해당 확률)
             shape: (문장 개수, 2)
    '''
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
            # shape: (batch 크기, 2)
            max_val_list, predict_list = torch.max(prob.data, -1)
            # 각각의 shape: (batch 크기,)
            is_malicious = ['악성 댓글' if predict else '악성 댓글 아님' for predict in predict_list]

            result = list(zip(is_malicious, max_val_list))
            result_list += result

    return result_list

