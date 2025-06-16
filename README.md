# 한국어 구조/의미 이해 Baseline
본 리포지토리는 '2025년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '한국어 구조/의미 이해'에 대한 베이스라인 모델의 추론을 재현하기 위한 코드를 포함하고 있습니다.  

추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

## 리포지토리 구조 (Repository Structure)
```
# 추론에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
└── test.py

# 추론에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
[
   {
      "id": "1",
      "input": {
         "category": "형태",
         "question_type": "다지선다",
         "question": "문제를 보고, 선지 중 알맞은 답을 선택하세요. \n<문제> 다음 접미사 '파'이 결합한 단어들 중 '파'의 의미가 다른 하나를 고르세요.\n<선지> (1) 소신파 (2) 육체파 (3) 지성파 (4) 고주파 \n정답 :"
      },
      "output": "4"
   }
]
```

## 실행 방법 (How to Run)
### 추론 (Inference)
```
(실제 코드는 25년 7월 중순에 업데이트 예정)
```


## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B)  
HyperCLOVAX-SEED-Text-Instruct-1.5B (https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)
