# AutoRAG 한국어 가이드

## 한국어 토크나이저

BM25 검색에서 한국어 토크나이저를 사용해야 정확한 검색이 가능합니다.

### 지원 토크나이저

| 토크나이저 | 라이브러리 | 설명 |
|-----------|-----------|------|
| `kiwi` | kiwipiepy | 빠르고 정확한 형태소 분석기 (권장) |
| `okt` | konlpy | Twitter 한국어 처리기 |
| `kkma` | konlpy | 꼬꼬마 형태소 분석기 |

### 설치

```bash
pip install "AutoRAG[ko]"
```

### 설정 예제

```yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: lexical_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 10
        modules:
          - module_type: bm25
            bm25_tokenizer: kiwi  # 한국어 토크나이저
```

### 하이브리드 검색

```yaml
- node_type: hybrid_retrieval
  strategy:
    metrics: [retrieval_f1, retrieval_recall]
  top_k: 10
  modules:
    - module_type: hybrid_rrf
      weight_range: (4, 80)
      bm25_tokenizer: kiwi  # 한국어 토크나이저 지정
```

---

## 한국어 BERTScore

생성 평가에서 한국어 BERTScore를 사용합니다.

### 설정 예제

```yaml
- node_type: generator
  strategy:
    metrics:
      - metric_name: rouge
      - metric_name: bert_score
        lang: ko                    # 한국어 BERTScore
      - metric_name: sem_score
        embedding_model: openai
  modules:
    - module_type: llama_index_llm
      llm: openai
      model: [gpt-4o-mini]
```

---

## 한국어 프롬프트 템플릿

### 기본 프롬프트

```yaml
- node_type: prompt_maker
  strategy:
    metrics: [bleu, meteor, rouge]
  modules:
    - module_type: fstring
      prompt: |
        주어진 문서를 참고하여 질문에 답하시오.

        문서:
        {retrieved_contents}

        질문: {query}

        답변:
```

### 상세 프롬프트

```yaml
- node_type: prompt_maker
  modules:
    - module_type: fstring
      prompt: |
        당신은 질문에 정확하게 답변하는 AI 어시스턴트입니다.
        아래 제공된 문서만을 참고하여 답변하세요.
        문서에 없는 내용은 답변하지 마세요.

        [참고 문서]
        {retrieved_contents}

        [질문]
        {query}

        [답변]
```

### 간결한 답변 프롬프트

```yaml
- module_type: fstring
  prompt: |
    문서: {retrieved_contents}

    위 문서를 바탕으로 다음 질문에 간결하게 답하세요.
    질문: {query}
    답변:
```

---

## 한국어 QA 데이터 생성

### 한국어 쿼리 생성

```python
from llama_index.llms.openai import OpenAI
from autorag.data.qa.query.llama_gen_query import factoid_query_gen

# 한국어 프롬프트로 LLM 설정
llm = OpenAI(
    model="gpt-4o-mini",
    system_prompt="당신은 한국어 질문을 생성하는 AI입니다. 모든 질문은 한국어로 작성하세요."
)

corpus = corpus.batch_apply(factoid_query_gen, llm=llm)
```

### 한국어 답변 생성

```python
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt

llm = OpenAI(
    model="gpt-4o-mini",
    system_prompt="당신은 한국어로 답변하는 AI입니다. 모든 답변은 한국어로 작성하세요."
)

corpus = corpus.batch_apply(make_basic_gen_gt, llm=llm)
```

### 한국어 필터링

```python
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based

# 한국어 "모르겠습니다" 필터링
corpus = corpus.filter(dontknow_filter_rule_based, lang="ko")
```

---

## 전체 한국어 설정 예제

```yaml
vectordb:
  - name: default
    db_type: chroma
    client_type: persistent
    embedding_model: openai
    collection_name: korean_docs
    path: ${PROJECT_DIR}/data/chroma

node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: lexical_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 10
        modules:
          - module_type: bm25
            bm25_tokenizer: kiwi

      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 10
        modules:
          - module_type: vectordb
            vectordb: default

      - node_type: hybrid_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 10
        modules:
          - module_type: hybrid_rrf
            weight_range: (4, 80)

  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: prompt_maker
        strategy:
          metrics: [bleu, meteor, rouge]
          generator_modules:
            - module_type: llama_index_llm
              llm: openai
              model: [gpt-4o-mini]
        modules:
          - module_type: fstring
            prompt: |
              주어진 문서를 참고하여 질문에 답하시오.

              문서: {retrieved_contents}

              질문: {query}

              답변:

      - node_type: generator
        strategy:
          metrics:
            - metric_name: rouge
            - metric_name: bert_score
              lang: ko
            - metric_name: sem_score
              embedding_model: openai
        modules:
          - module_type: llama_index_llm
            llm: openai
            model: [gpt-4o-mini]
```

---

## 한국어 데이터 전처리 팁

### 1. 문서 정규화

한국어 문서는 띄어쓰기, 특수문자 처리가 중요합니다.

```python
import re

def normalize_korean_text(text):
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 특수문자 정규화
    text = re.sub(r'[^\w\s.,!?가-힣]', '', text)
    return text.strip()
```

### 2. 청킹 설정

한국어는 영어보다 토큰 수가 많으므로 청크 크기를 조정합니다.

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: Token
    chunk_size: 512       # 영어 대비 작게 설정
    chunk_overlap: 64
```

### 3. 형태소 분석 기반 청킹

Sentence 청킹은 한국어에서도 잘 작동합니다.

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: Sentence
    chunk_size: 1024
```

---

## 문제 해결

### kiwi 설치 오류

```bash
# macOS
brew install cmake
pip install kiwipiepy

# Linux
sudo apt-get install cmake
pip install kiwipiepy
```

### konlpy 설치 오류

```bash
# Java 설치 필요
sudo apt-get install default-jdk

# konlpy 설치
pip install konlpy
```

### 한국어 BERTScore 느림

BERTScore는 GPU 사용 시 빨라집니다.

```bash
pip install "AutoRAG[gpu]"
```

### 인코딩 문제

Parquet 파일 저장 시 UTF-8 인코딩을 확인합니다.

```python
df.to_parquet('data.parquet', engine='pyarrow')
```
