from preprocessing import preprocess_text
from preparing_nlp import prepare_nlp
from modeling import classify

import time

def classify_comment(text_list):
    after_preprocess = preprocess_text(text_list)
    after_prepare = prepare_nlp(after_preprocess)
    after_classify = classify(after_prepare)

    result_list = {}

    for i in range(len(text_list)):
        result = {}
        result['text'] = text_list[i]
        result['악성 댓글 여부'] = after_classify[i][0]
        result['확률'] = '{:.2f}%'.format(after_classify[i][1].item() * 100)

        result_list[f'idx_{i}'] = result

    return result_list

if __name__ == '__main__':
    start = time.time()
    result = classify_comment(['도전을 안 하는 것만큼 무의미한 것은 없어',
                               '야 이 자식아 너 뭐하냐 죽여버린다'])
    print(result)
    end = time.time()

    print(f"{end - start:.3f} sec")