# 국어문법 질의응답 Baseline
본 리포지토리는 '2025년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '국어문법 질의응답'에 대한 베이스라인 모델의 추론을 재현하기 위한 코드를 포함하고 있습니다.  

추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

### Baseline
|           Model           | Accuracy(%) |
| :-----------------------: | :---------: |
|        **Qwen3-8B**        |    0.548    |
| **HyperCLOVAX Text 1.5B** |    0.378    |

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
      "id": "nikluge-2025-국어_문법_질의응답-dev-1",
      "input": {
         "category": "담화",
         "question_type": "다지선다",
         "question": "문제를 보고, 알맞은 답을 쓰세요.\n<문제> 다음 글에 이어지는 문장으로 알맞은 것을 고르세요.\n 현대인들이 먹는 모든 음식이 환경호르몬에 오염돼 있는 것으로 조사됐다. 이같은 사실은 미국 뉴욕주립대 보건과학센터 연구팀이 미국 전역의 슈퍼마켓에서 수거한 식료품을 분석한 결과 드러났다.\n(1) 다이옥신은 발암물질이자 환경호르몬으로 추정되는 화학 물질이다.\n(2) 연구팀이 작성한 보고서에 따르면 거의 모든 시료품에서 다이옥신이 검출됐다.\n(3) 주로 쓰레기 소각장에서 다이옥신과 같은 환경호르몬이 발생한다.\n(4) 환경호르몬 물질에 대한 과학적인 검증과 대처가 반드시 있어야 한다.\n "
      },
      "output": "2"
   }
]
```

## 실행 방법 (How to Run)
### 추론 (Inference)
```
CUDA_VISIBLE_DEVICES=0 python -m run.test \
    --input resource/data/sample.json \
    --output result.json \
    --model_id Qwen/Qwen3-8B \
    --debug True
```


## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B)  
HyperCLOVAX-SEED-Text-Instruct-1.5B (https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)
