# AI
AI_Project
model들은 임시로 넣은 모델이라서 변경될 수 있음

[프론트: Java ( React )]
↓
[메인 백엔드: Java Spring]
↓
1. 게시글 저장
2. 사용자 인증
3. 댓글 등록 요청d
   ↓
   ▶ Python 욕설 탐지 서버 (Flask) 호출
   ↓
   결과 받아서 처리 (차단/숨김/경고 등)

```js(서버 실행방법)

   pip install -r requirements.txt     
   python app.py

```

```js(python api호출로직)

    // spring 예제
    RestTemplate restTemplate = new RestTemplate();
    HttpHeaders headers = new HttpHeaders();
    headers.setContentType(MediaType.APPLICATION_JSON);
    
    Map<String, String> payload = Map.of("text", "너 진짜 왜 그래");
    HttpEntity<Map<String, String>> entity = new HttpEntity<>(payload, headers);
    
    String url = "http://localhost:8000/check-abuse";
    ResponseEntity<String> response = restTemplate.postForEntity(url, entity, String.class);
    
    System.out.println(response.getBody());


    // React 예제 (axios 사용)
    import axios from 'axios';
    
    const checkAbuse = async (text) => {
      const response = await axios.post('http://localhost:8000/check-abuse', {
        text: text,
      });
      return response.data;
    };

```