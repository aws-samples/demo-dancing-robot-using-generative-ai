import numpy as np
import time
from lambda_function import *
import pprint

def load_event():
    json_data = {
        "userId": "user1234",
        "requestId": "test1234",
        "request_time": "2023-10-08 18:01:45",        
        "type": "text",
        "text": "안녕하세요",
        "mbti": "ISTP",
        "has_description": True,
    }
    return {"body": json.dumps(json_data)}


pprint.pprint(load_event())
def _test_extract_text_from_tags():
    text = """
<score>5</score>

<description>
이 문장은 주인에게 충성스럽고 활동적인 성격을 가진 캐릭터에게 매우 긍정적으로 받아들여질 것입니다. '나랑놀자'라는 말은 주인이 애완동물과 함께 시간을 보내고 싶어한다는 의미로, 충성스럽고 전투적이며 부지런한 성격의 캐릭터는 주인의 요청에 기쁘게 응할 것입니다. 또한 게으른 것을 싫어하는 성격 때문에 활동적인 놀이를 환영할 것입니다. 고집이 강더라도 주인에 대한 충성심 때문에 거절하지 않을 것입니다.
</description>
"""
    tag = "score"
    result = extract_text_from_tags(text, tag)[0]
    score = int(result[0])
    print(score)

def main():
    start = time.time()

    # load samples
    event = load_event()

    # run
    _test_extract_text_from_tags()
    results = lambda_handler(event, "")
    pprint.pprint(results)

    # results
    print(results['statusCode'])
    print(results['body'])

    print('Elapsed time: %0.2fs' % (time.time()-start))   

if __name__ == '__main__':
    main()
