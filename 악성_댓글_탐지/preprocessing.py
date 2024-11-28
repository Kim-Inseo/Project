from konlpy.tag import Okt
from transformers import T5ForConditionalGeneration, T5Tokenizer

corr_model = T5ForConditionalGeneration.from_pretrained('j5ng/et5-typos-corrector')
tokenizer = T5Tokenizer.from_pretrained('j5ng/et5-typos-corrector')

okt = Okt()
stopwords = ['하다']

# 맞춤법 수정 함수
def spell_correction(text_list, corr_model, tokenizer):
    '''
    :param text_list: 1차원 list(각 원소는 한국어 문장)
    :param corr_model: 사전 훈련된 모델
    :param tokenizer: 사전 훈련된 모델
    :return: 1차원 list(각 원소는 맞춤법이 수정된 한국어 문장)
    '''
    output_text_list = []

    for text in text_list:
        input_encoding = tokenizer("맞춤법을 고쳐주세요: " + text, return_tensors="pt")

        input_ids = input_encoding.input_ids
        attention_mask = input_encoding.attention_mask

        output_encoding = corr_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )

        output_text = tokenizer.decode(output_encoding[0], skip_special_tokens=True)
        output_text_list.append(output_text)

    return output_text_list

# 토큰화 함수 작성
def tokenize(text_list, analyzer, stopwords):
    '''
    :param text_list: 1차원 list(각 원소는 맞춤법이 수정된 한국어 문장)
    :param analyzer: 한국어 형태소 분석기(Okt)
    :param stopwords: 한국어 불용어 list
    :return: 2차원 list
            (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)
    '''
    output_tokens_list = []

    for text in text_list:
        pos_token = analyzer.pos(text, stem=True, norm=True)

        after_pos = []
        exclude_tag_list = ['Josa', 'PreEomi', 'Eomi', 'Punctuation', 'Foreign']

        for token, tag in pos_token:
            if tag not in exclude_tag_list:
                after_pos.append(token)

        tokens = [word for word in after_pos if word not in stopwords]
        output_tokens_list.append(tokens)

    return output_tokens_list

# 전처리 작업 수행
def preprocess_text(text_list):
    '''
    :param text_list: 1차원 list(각 원소는 한국어 문장)
    :return: 2차원 list
            (첫 번째 차원에 문장, 두 번째 차원에 각 문장별 단어 토큰)

    spell_correction(), tokenize()를 한 함수로 묶음
    '''
    corr_text_list = spell_correction(text_list, corr_model, tokenizer)
    output_tokens_list = tokenize(corr_text_list, okt, stopwords)

    return output_tokens_list



