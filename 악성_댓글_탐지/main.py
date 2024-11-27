from fastapi import FastAPI
from comment import Item

from preprocessing import preprocess_text
from preparing_nlp import prepare_nlp
from modeling import classify

app = FastAPI()

@app.post('/items/')
async def classify_comment(item: Item):
    after_preprocess = preprocess_text(item.text)
    after_prepare = prepare_nlp(after_preprocess)
    after_classify = classify(after_prepare)

    result_list = {}

    for i in range(len(item.text)):
        result = {}
        result['text'] = item.text[i]
        result['악성 댓글 여부'] = after_classify[i][0]
        result['확률'] = '{:.2f}%'.format(after_classify[i][1].item() * 100)

        result_list[f'idx_{i}'] = result

    return result_list