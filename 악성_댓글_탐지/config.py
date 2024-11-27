class ConfigModel:
    def __init__(self, model_path, var_models_path):
        self.model_path = model_path
        self.var_models_path = var_models_path

class ConfigUtils:
    def __init__(self, tokenizer_path, var_utils_path, fasttext_path):
        self.tokenizer_path = tokenizer_path
        self.var_utils_path = var_utils_path
        self.fasttext_path = fasttext_path