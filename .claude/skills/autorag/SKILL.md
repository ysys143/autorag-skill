---
name: autorag
description: AutoRAG는 RAG(Retrieval-Augmented Generation) 파이프라인을 자동으로 최적화하는 AutoML 도구입니다. 문서 파싱, 청킹, QA 데이터 생성, RAG 파이프라인 최적화, 배포를 지원합니다. 사용자가 RAG 시스템을 구축하거나 최적화할 때 이 스킬을 활용하세요.
license: Apache-2.0
compatibility: Python 3.10+, OpenAI API 키 또는 로컬 LLM 필요
metadata:
  category: ml-ops
  languages: [python]
  frameworks: [llamaindex, langchain]
---

# AutoRAG

AutoRAG는 사용자의 데이터에 맞는 최적의 RAG 파이프라인을 자동으로 탐색하는 AutoML 도구입니다.

## 언제 이 스킬을 사용하는가

다음 상황에서 이 스킬을 활용하세요:

- RAG 시스템을 처음부터 구축할 때
- 기존 RAG 파이프라인의 성능을 개선하고 싶을 때
- 다양한 검색/생성 모듈 조합을 평가하고 싶을 때
- 최적화된 RAG 파이프라인을 API로 배포하고 싶을 때
- 문서를 파싱하고 청킹하여 RAG용 데이터를 준비할 때

## 핵심 워크플로우

```
[원본 문서] → [파싱] → [청킹] → [QA 데이터 생성] → [RAG 최적화] → [배포]
```

### 1단계: 데이터 준비

**문서 파싱**
```python
from autorag.parser import Parser

parser = Parser(data_path_glob="documents/*.pdf")
parser.start_parsing("parse_config.yaml")
```

**문서 청킹**
```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet("parsed.parquet")
chunker.start_chunking("chunk_config.yaml")
```

**QA 데이터 생성**
```python
from autorag.data.qa.schema import Raw, Corpus
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt
from autorag.data.qa.sample import random_single_hop

corpus = Corpus(corpus_df, raw)
qa_data = (
    corpus
    .sample(random_single_hop, n=100)
    .make_retrieval_gt_contents()
    .batch_apply(factoid_query_gen, llm=llm)
    .batch_apply(make_basic_gen_gt, llm=llm)
)
qa_data.to_parquet('./qa.parquet', './corpus.parquet')
```

### 2단계: RAG 최적화

**Python**
```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path='qa.parquet',
    corpus_data_path='corpus.parquet'
)
evaluator.start_trial('config.yaml')
```

**CLI**
```bash
autorag evaluate \
    --config config.yaml \
    --qa_data_path qa.parquet \
    --corpus_data_path corpus.parquet
```

### 3단계: 배포

**코드에서 사용**
```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('./project/0')
result = runner.run('What is machine learning?')
```

**API 서버**
```bash
autorag run_api --trial_dir ./project/0 --port 8000
```

**웹 인터페이스**
```bash
autorag run_web --trial_path ./project/0
```

## 빠른 시작 설정 파일

가장 간단한 RAG 최적화 설정:

```yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 3
        modules:
          - module_type: vectordb
            vectordb: default

  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: prompt_maker
        strategy:
          metrics: [bleu, meteor, rouge]
        modules:
          - module_type: fstring
            prompt: "Question: {query}\nContext: {retrieved_contents}\nAnswer:"

      - node_type: generator
        strategy:
          metrics: [bleu, rouge]
        modules:
          - module_type: llama_index_llm
            llm: openai
            model: [gpt-4o-mini]
```

## CLI 명령어 요약

| 명령어 | 설명 |
|-------|------|
| `autorag evaluate` | RAG 파이프라인 최적화 실행 |
| `autorag validate` | 설정 파일 검증 |
| `autorag run_api` | API 서버 실행 |
| `autorag run_web` | 웹 인터페이스 실행 |
| `autorag dashboard` | 대시보드 실행 |
| `autorag extract_best_config` | 최적 설정 추출 |
| `autorag restart_evaluate` | 중단된 최적화 재개 |

상세 옵션은 `references/CLI_REFERENCE.md` 참조

## 핵심 Python API

| 클래스 | 용도 |
|-------|------|
| `Evaluator` | RAG 파이프라인 최적화 |
| `Parser` | 문서 파싱 |
| `Chunker` | 문서 청킹 |
| `Runner` | 최적화된 파이프라인 실행 |
| `ApiRunner` | API 서버 실행 |

상세 API는 `references/API_REFERENCE.md` 참조

## 노드 타입

| 노드 | 설명 | 단계 |
|-----|------|------|
| `query_expansion` | 쿼리 확장/분해 | 검색 전 |
| `lexical_retrieval` | BM25 검색 | 검색 |
| `semantic_retrieval` | 벡터 검색 | 검색 |
| `hybrid_retrieval` | 하이브리드 검색 | 검색 |
| `passage_reranker` | 재순위 | 검색 후 |
| `passage_filter` | 필터링 | 검색 후 |
| `passage_compressor` | 압축 | 검색 후 |
| `prompt_maker` | 프롬프트 구성 | 생성 전 |
| `generator` | 답변 생성 | 생성 |

## 평가 지표

**검색 지표**: `retrieval_f1`, `retrieval_recall`, `retrieval_precision`, `retrieval_ndcg`, `retrieval_mrr`, `retrieval_map`

**생성 지표**: `bleu`, `meteor`, `rouge`, `sem_score`, `bert_score`, `g_eval`

## 데이터 스키마

**QA 데이터 (qa.parquet)**
- `qid`: 질문 ID
- `query`: 질문 텍스트
- `retrieval_gt`: 검색 정답 문서 ID 목록
- `generation_gt`: 생성 정답 텍스트 목록

**Corpus 데이터 (corpus.parquet)**
- `doc_id`: 문서 ID
- `contents`: 문서 내용
- `metadata`: 메타데이터 (선택)

## 한국어 지원

한국어 RAG 시스템 구축 시 `references/KOREAN_GUIDE.md` 참조

주요 설정:
- BM25 토크나이저: `kiwi`, `okt`, `kkma`
- BERTScore: `lang: ko`
- 프롬프트: 한국어 템플릿 사용

## 문제 해결

**설치 오류**
```bash
pip install "AutoRAG[gpu]"  # GPU 버전
pip install "AutoRAG[ko]"   # 한국어 지원
pip install "AutoRAG[all]"  # 전체 기능
```

**OpenAI API 키 설정**
```bash
export OPENAI_API_KEY="your-api-key"
```

**메모리 부족**
- `skip_validation=True` 옵션으로 검증 건너뛰기
- `full_ingest=False`로 필요한 문서만 인덱싱

**중단된 최적화 재개**
```bash
autorag restart_evaluate --trial_path ./project/0
```

## 추가 자료

- 설정 파일 상세: `references/CONFIG_REFERENCE.md`
- CLI 상세: `references/CLI_REFERENCE.md`
- Python API 상세: `references/API_REFERENCE.md`
- 한국어 가이드: `references/KOREAN_GUIDE.md`
- 설정 템플릿: `assets/` 디렉토리
