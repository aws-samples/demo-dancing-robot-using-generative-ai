import requests

# API Gateway 엔드포인트 URL
base_url = 'https://2zl6szvhq5.execute-api.ap-northeast-2.amazonaws.com/dev'
api_gateway_url = base_url + '/score'

# API Gateway에 전달할 데이터
data = {
    "userId": "user1234",
    "requestId": "test1234",
    "request_time": "2023-10-08 18:01:45",
    "type": "text",
    "text": "산책가자",
    "mbti": "ISTP",
    "has_description": True,
}

# HTTP POST 요청 보내기
response = requests.post(api_gateway_url, json=data)

# 응답 처리
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print(f'Error: {response.status_code} - {response.text}')