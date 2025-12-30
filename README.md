# AutoRAG Claude Code Skill

AutoRAG를 활용한 RAG 파이프라인 구축 및 최적화를 지원하는 Claude Code Skill입니다. Claude Code에서 `/autorag` 명령으로 호출하여 RAG 시스템 구축에 필요한 전문 지식과 코드 생성을 지원받을 수 있습니다.

## 개요

AutoRAG는 사용자의 데이터에 맞는 최적의 RAG(Retrieval-Augmented Generation) 파이프라인을 자동으로 탐색하는 AutoML 도구입니다. 이 Skill은 Claude Code가 AutoRAG의 복잡한 설정과 API를 정확하게 이해하고 활용할 수 있도록 구조화된 레퍼런스 문서와 예제를 제공합니다.

## 설치

### 1. Skill 디렉토리 복사

이 저장소의 `.claude/skills/autorag` 디렉토리를 프로젝트의 `.claude/skills/` 경로에 복사합니다.

```bash
# 프로젝트 디렉토리에서
mkdir -p .claude/skills
cp -r /path/to/autorag-skill/.claude/skills/autorag .claude/skills/
```

또는 전역으로 설치하려면 `~/.claude/skills/` 경로에 복사합니다.

```bash
mkdir -p ~/.claude/skills
cp -r /path/to/autorag-skill/.claude/skills/autorag ~/.claude/skills/
```

### 2. AutoRAG 패키지 설치

Python 환경에 AutoRAG 패키지를 설치합니다.

```bash
# 기본 설치
pip install AutoRAG

# GPU 지원
pip install "AutoRAG[gpu]"

# 한국어 지원
pip install "AutoRAG[ko]"

# 전체 기능
pip install "AutoRAG[all]"
```

### 3. API 키 설정

```bash
export OPENAI_API_KEY="your-api-key"
```

## Skill 구조

```
.claude/skills/autorag/
├── SKILL.md                    # 메인 Skill 정의 파일
├── references/
│   ├── CLI_REFERENCE.md        # CLI 명령어 레퍼런스
│   ├── CONFIG_REFERENCE.md     # 설정 파일 레퍼런스
│   ├── API_REFERENCE.md        # Python API 레퍼런스
│   └── KOREAN_GUIDE.md         # 한국어 RAG 구축 가이드
└── assets/
    ├── simple_config.yaml      # 기본 설정 템플릿
    ├── korean_config.yaml      # 한국어 설정 템플릿
    └── full_config.yaml        # 전체 옵션 설정 템플릿
```

## 사용법

Claude Code에서 AutoRAG 관련 작업을 요청하면 이 Skill이 자동으로 활성화됩니다. 다음과 같은 작업에서 활용됩니다.

### 지원 작업

- RAG 시스템 설계 및 구현
- 문서 파싱 및 청킹 파이프라인 구축
- QA 데이터셋 생성
- RAG 파이프라인 최적화 설정
- 최적화된 파이프라인 API 배포
- 한국어 RAG 시스템 구축

### 예시 대화

```
사용자: AutoRAG로 PDF 문서 기반 RAG 시스템을 구축하고 싶어요.

Claude: [SKILL.md의 워크플로우와 API_REFERENCE.md를 참조하여]
        문서 파싱부터 RAG 배포까지 단계별 코드와 설정을 제공
```

```
사용자: 한국어 문서를 위한 BM25 검색 설정을 알려주세요.

Claude: [KOREAN_GUIDE.md를 참조하여]
        kiwi 토크나이저를 사용한 한국어 BM25 설정 제공
```

## 핵심 워크플로우

AutoRAG를 활용한 RAG 시스템 구축은 다음 단계로 진행됩니다.

```
[원본 문서] → [파싱] → [청킹] → [QA 데이터 생성] → [RAG 최적화] → [배포]
```

### Phase 1: 데이터 준비

문서 파싱과 청킹을 통해 RAG용 Corpus를 준비합니다.

```python
from autorag.parser import Parser
from autorag.chunker import Chunker

# 문서 파싱
parser = Parser(data_path_glob="documents/*.pdf")
parser.start_parsing("parse_config.yaml")

# 청킹
chunker = Chunker.from_parquet("parsed.parquet")
chunker.start_chunking("chunk_config.yaml")
```

### Phase 2: QA 데이터 생성

최적화를 위한 QA 데이터셋을 생성합니다.

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

### Phase 3: RAG 최적화

다양한 검색/생성 모듈 조합을 평가하여 최적의 파이프라인을 탐색합니다.

```bash
autorag evaluate \
    --config config.yaml \
    --qa_data_path qa.parquet \
    --corpus_data_path corpus.parquet
```

### Phase 4: 배포

최적화된 파이프라인을 API 서버로 배포합니다.

```bash
autorag run_api --trial_dir ./project/0 --port 8000
```

## 설정 템플릿

### 기본 설정 (simple_config.yaml)

OpenAI API 기반의 간단한 RAG 파이프라인입니다. 빠른 프로토타이핑에 적합합니다.

### 한국어 설정 (korean_config.yaml)

한국어 토크나이저(kiwi)와 한국어 BERTScore를 포함한 설정입니다. 한국어 문서 기반 RAG 시스템 구축에 사용합니다.

### 전체 설정 (full_config.yaml)

쿼리 확장, 다양한 검색 방식, 문서 필터링/압축, 다중 생성 모델 등 모든 노드 타입을 포함한 설정입니다. 최적의 파이프라인 탐색에 사용하지만 실행 시간이 오래 걸립니다.

## 참조 문서

| 문서 | 설명 |
|------|------|
| `SKILL.md` | Skill 메인 정의, 핵심 워크플로우, 빠른 시작 가이드 |
| `CLI_REFERENCE.md` | CLI 명령어 상세 옵션 및 예제 |
| `CONFIG_REFERENCE.md` | 설정 파일 구조, 노드 타입별 설정, 평가 지표 |
| `API_REFERENCE.md` | Python API 클래스 및 함수 레퍼런스 |
| `KOREAN_GUIDE.md` | 한국어 토크나이저, 프롬프트, 데이터 전처리 가이드 |

## 지원 기능

### 노드 타입

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

### 평가 지표

**검색 지표**: `retrieval_f1`, `retrieval_recall`, `retrieval_precision`, `retrieval_ndcg`, `retrieval_mrr`, `retrieval_map`

**생성 지표**: `bleu`, `meteor`, `rouge`, `sem_score`, `bert_score`, `g_eval`

### 지원 벡터DB

Chroma, Milvus, Weaviate, Pinecone, Qdrant, Couchbase

### 지원 LLM

OpenAI, OpenAI-compatible API, Ollama, HuggingFace, AWS Bedrock

## 요구 사항

- Python 3.10 이상
- OpenAI API 키 또는 로컬 LLM
- 한국어 지원 시 kiwipiepy 또는 konlpy

## 라이센스

Apache-2.0

## 관련 링크

- [AutoRAG GitHub](https://github.com/Marker-Inc-Korea/AutoRAG)
- [AutoRAG 공식 문서](https://docs.auto-rag.com/)
