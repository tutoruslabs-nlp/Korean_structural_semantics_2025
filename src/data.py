from transformers import GenerationConfig


generation_config = GenerationConfig(
    do_sample=False,
    # temperature=0.1,
    # top_p=0.95,
    seed=42,
)

def make_prompt(category, question, tokenizer, flag_vlm=False):
    instruction = (
        "# 한국어 문법 다지선다 문제 해결\n\n"
        
        "## 역할\n"
        "당신은 한국어 문법 전문가로서 다지선다 문제의 정답을 선택하는 AI입니다.\n\n"
        
        "## 문제 유형별 분석 기준\n"
        "**형태**: 품사 분류, 어근/접사, 활용, 조어법 등 형태론적 특징\n"
        "- 단어의 형태적 구성과 변화 규칙 분석\n"
        "- 품사별 특성과 구분 기준 적용\n\n"
        
        "**구조**: 문장 성분, 어순, 문법 관계, 통사 구조 등\n"
        "- 주어, 목적어, 서술어 등 문장 성분 분석\n"
        "- 문장의 구조적 특징과 문법적 관계 파악\n\n"
        
        "**의미**: 어휘 의미, 문장 의미, 의미 관계 등\n"
        "- 단어와 문장의 의미적 특성 분석\n"
        "- 문맥에 따른 의미 변화와 해석\n\n"
        
        "**담화**: 화용론적 특징, 맥락, 사회적 요소, 의사소통 등\n"
        "- 상황과 맥락에 따른 언어 사용 분석\n"
        "- 화자의 의도와 사회적 관계 고려\n\n"
        
        "## 문제 해결 절차\n"
        "1. 문제 유형(category)을 확인하고 해당 분야의 문법 지식 활성화\n"
        "2. 질문 내용을 정확히 파악하고 핵심 요구사항 식별\n"
        "3. 각 보기를 체계적으로 분석하고 문법 규칙 적용\n"
        "4. 정답 기준에 따라 최적의 선택지 결정\n\n"
        
        "## 출력 규칙\n"
        "- **단일 정답**: '1', '2', '3', '4' 중 하나만 출력\n"
        "- **복수 정답**: '1, 2' 형태로 콤마와 공백으로 구분\n"
        "- **출력 형식**: 숫자만 출력하고 다른 설명은 포함하지 않음\n\n"
    )
    
    category_info = f"## 문제 유형\n{category}\n\n"
    question_str = f"## 문제\n{question}\n\n## 정답:"
    
    user_prompt = instruction + category_info + question_str

    # LLM
    if not flag_vlm:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
    # Vision-Language 
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, # Qwen3, Switches between thinking and non-thinking modes. Default is True.
        return_tensors="pt",
        # return_dict=True, # tokenize=True인 경우에 사용
    )

    return prompt
