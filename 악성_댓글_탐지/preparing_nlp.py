import numpy as np
import json
from gensim.models import FastText
from config import ConfigUtils

def padding(text_list):
    '''
    :param text_list: 2차원 list
                      (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)
    :return: 2차원 list
             (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)
             shape: (문장 개수, sequence 길이)

    padding을 수행하는 함수
    지정된 max_len보다 큰 sequence가 들어오면 초과된 길이의 앞 부분을 잘라내고,
    지정된 max_len보다 작은 sequence가 들어오면 모자란 길이의 앞 부분에 <pad>를 추가
    '''
    config_utils = ConfigUtils(tokenizer_path='./utils/tokenizer.pickle',
                               var_utils_path='./utils/var_utils.json',
                               fasttext_path='./utils/fastText_pretrained.model')
    var_utils_path = config_utils.var_utils_path
    with open(var_utils_path, 'r') as f:
        var_utils_dict = json.load(f)
    max_len = var_utils_dict['max_len']

    pad_text_list = []

    for text_tokens in text_list:
        if len(text_tokens) >= max_len:
            pad_text = text_tokens[len(text_tokens) - max_len:]
        else:
            pad_text = ['<pad>' for _ in range((max_len - len(text_tokens)))] + text_tokens
        pad_text_list.append(pad_text)

    return pad_text_list

def vectorization(text_list):
    '''
    :param text_list: 2차원 list
                      (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)
                      shape: (문장 개수, sequence 길이)
    :return: 3차원 list
             (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰,
             세 번째 차원에 단어 토큰에 대응하는 FastText 임베딩 벡터)
             shape: (문장 개수, sequence 길이, 임베딩 벡터 크기)

    사전 훈련된 FastText 모델을 이용해 단어 토큰을 임베딩 벡터로 변환
    FastText 모델은 OOV에 해당하는 단어도 임베딩 벡터로 바꿀 수 있다.
    '''
    config_utils = ConfigUtils(tokenizer_path='./utils/tokenizer.pickle',
                               var_utils_path='./utils/var_utils.json',
                               fasttext_path='./utils/fastText_pretrained.model')
    fasttext_path = config_utils.fasttext_path

    fastText = FastText.load(fasttext_path)
    text_vec_list = []

    vec_size = fastText.vector_size

    for text_tokens in text_list:
        text_vec = []
        for token in text_tokens:
            if token == '<pad>':
                text_vec.append(np.zeros(vec_size))
            else:
                text_vec.append(fastText.wv.get_vector(token))
        text_vec_list.append(text_vec)

    return text_vec_list


def prepare_nlp(text_list):
    '''
    :param text_list: 2차원 list
                      (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)
    :return: 3차원 list
             (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰,
             세 번째 차원에 단어 토큰에 대응하는 FastText 임베딩 벡터)

    padding()과 vectorization() 함수를 한 함수로 묶음
    '''
    pad_text_list = padding(text_list)
    vector_text_list = vectorization(pad_text_list)

    return vector_text_list


