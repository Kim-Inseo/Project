import numpy as np
import json
from gensim.models import FastText
from config import ConfigUtils

def padding(text_list):
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
    pad_text_list = padding(text_list)
    vector_text_list = vectorization(pad_text_list)

    return vector_text_list


