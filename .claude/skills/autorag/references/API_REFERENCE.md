# AutoRAG Python API 레퍼런스

## Evaluator

RAG 파이프라인 최적화 엔진입니다.

```python
from autorag.evaluator import Evaluator

# 초기화
evaluator = Evaluator(
    qa_data_path: str,           # QA 데이터 Parquet 경로
    corpus_data_path: str,       # Corpus 데이터 Parquet 경로
    project_dir: str = None      # 프로젝트 디렉토리 (기본: 현재 디렉토리)
)

# 최적화 실행
evaluator.start_trial(
    yaml_path: str,              # 설정 YAML 경로
    skip_validation: bool = False,  # 검증 건너뛰기
    full_ingest: bool = True     # 전체 Corpus 인제스트
)

# 중단된 최적화 재개
evaluator.restart_trial(trial_path: str)
```

**예제**
```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path='data/qa.parquet',
    corpus_data_path='data/corpus.parquet',
    project_dir='./my_project'
)

# 최적화 실행
evaluator.start_trial('config.yaml')

# 결과 확인: ./my_project/0/summary.csv
```

---

## Parser

문서를 파싱하여 텍스트로 변환합니다.

```python
from autorag.parser import Parser

# 초기화
parser = Parser(
    data_path_glob: str          # 파일 경로 패턴 (예: "docs/*.pdf")
)

# 파싱 실행
parser.start_parsing(
    yaml_path: str,              # 파싱 설정 YAML 경로
    all_files: bool = False      # True: 모든 파일, False: 샘플
)
```

**파싱 설정 예제** (`parse_config.yaml`)
```yaml
modules:
  - module_type: langchain_parse
    file_type: pdf
    parse_method: pdfminer
```

**예제**
```python
from autorag.parser import Parser

parser = Parser(data_path_glob="documents/*.pdf")
parser.start_parsing("parse_config.yaml")

# 결과: parsed.parquet
```

---

## Chunker

파싱된 문서를 청크로 분할합니다.

```python
from autorag.chunker import Chunker

# 초기화 (Parquet 파일에서)
chunker = Chunker.from_parquet(
    parsed_data_path: str,       # 파싱된 데이터 Parquet 경로
    project_dir: str = None      # 프로젝트 디렉토리
)

# 청킹 실행
chunker.start_chunking(yaml_path: str)
```

**청킹 설정 예제** (`chunk_config.yaml`)
```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: Token
    chunk_size: 1024
    chunk_overlap: 128
```

**예제**
```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet("parsed.parquet")
chunker.start_chunking("chunk_config.yaml")

# 결과: corpus.parquet
```

---

## Validator

설정과 데이터를 검증합니다.

```python
from autorag.validator import Validator

# 초기화
validator = Validator(
    qa_data_path: str,           # QA 데이터 경로
    corpus_data_path: str        # Corpus 데이터 경로
)

# 검증 실행
validator.validate(
    yaml_path: str,              # 설정 YAML 경로
    qa_cnt: int = 5,             # 검증에 사용할 QA 샘플 수
    random_state: int = 42       # 랜덤 시드
)
```

---

## Runner

최적화된 파이프라인을 실행합니다.

```python
from autorag.deploy import Runner

# 트리얼에서 로드
runner = Runner.from_trial_folder(trial_path: str)

# YAML에서 로드
runner = Runner.from_yaml(
    yaml_path: str,
    project_dir: str
)

# 실행
result = runner.run(
    query: str,                  # 질문
    result_column: str = "generated_texts"  # 반환할 컬럼
)
```

**반환값**
```python
{
    'answer': str,               # 생성된 답변
    'retrieved_contents': List,  # 검색된 문서 내용
    'retrieved_ids': List,       # 검색된 문서 ID
    'retrieve_scores': List      # 검색 점수
}
```

**예제**
```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder('./project/0')
result = runner.run('What is machine learning?')

print(result['answer'])
print(result['retrieved_contents'])
```

---

## ApiRunner

최적화된 파이프라인을 API 서버로 실행합니다.

```python
from autorag.deploy import ApiRunner
import nest_asyncio

# 트리얼에서 로드
runner = ApiRunner.from_trial_folder(trial_path: str)

# YAML에서 로드
runner = ApiRunner.from_yaml(
    yaml_path: str,
    project_dir: str
)

# API 서버 실행
nest_asyncio.apply()  # Jupyter 환경에서 필요
runner.run_api_server(
    host: str = '0.0.0.0',
    port: int = 8000,
    remote: bool = False         # ngrok 터널 생성
)
```

**예제**
```python
from autorag.deploy import ApiRunner
import nest_asyncio

nest_asyncio.apply()

runner = ApiRunner.from_trial_folder('./project/0')
runner.run_api_server(host='0.0.0.0', port=8000)
```

---

## QA 데이터 생성 API

### Raw 클래스

파싱된 원본 문서를 관리합니다.

```python
from autorag.data.qa.schema import Raw
import pandas as pd

raw_df = pd.read_parquet("parsed.parquet")
raw = Raw(raw_df)

# 변환 적용
raw = raw.batch_apply(func, **kwargs)
raw = raw.map(func, **kwargs)

# 청킹으로 변환
corpus = raw.chunk(module_name: str, **params)
```

### Corpus 클래스

청킹된 문서와 QA 생성을 관리합니다.

```python
from autorag.data.qa.schema import Corpus

corpus = Corpus(corpus_df, raw)

# 샘플 추출
corpus = corpus.sample(sampler_func, n=100)

# 변환 적용
corpus = corpus.map(func, **kwargs)
corpus = corpus.batch_apply(func, **kwargs)

# 필터링
corpus = corpus.filter(filter_func, **kwargs)

# retrieval GT 생성
corpus = corpus.make_retrieval_gt_contents()

# 저장
corpus.to_parquet(qa_path: str, corpus_path: str)
```

### QA 생성 함수

**쿼리 생성**
```python
from autorag.data.qa.query.llama_gen_query import factoid_query_gen

corpus = corpus.batch_apply(factoid_query_gen, llm=llm)
```

**답변 생성**
```python
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
    make_concise_gen_gt
)

corpus = corpus.batch_apply(make_basic_gen_gt, llm=llm)
corpus = corpus.batch_apply(make_concise_gen_gt, llm=llm)
```

**필터링**
```python
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based

corpus = corpus.filter(dontknow_filter_rule_based, lang="en")
```

**샘플링**
```python
from autorag.data.qa.sample import random_single_hop

corpus = corpus.sample(random_single_hop, n=100)
```

### 전체 예제

```python
import pandas as pd
from llama_index.llms.openai import OpenAI
from autorag.data.qa.schema import Raw, Corpus
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based

# LLM 초기화
llm = OpenAI(model="gpt-4o-mini")

# 데이터 로드
raw_df = pd.read_parquet("parsed.parquet")
corpus_df = pd.read_parquet("corpus.parquet")

raw = Raw(raw_df)
corpus = Corpus(corpus_df, raw)

# QA 생성 파이프라인
qa_data = (
    corpus
    .sample(random_single_hop, n=100)
    .map(lambda df: df.reset_index(drop=True))
    .make_retrieval_gt_contents()
    .batch_apply(factoid_query_gen, llm=llm)
    .batch_apply(make_basic_gen_gt, llm=llm)
    .filter(dontknow_filter_rule_based, lang="en")
)

# 저장
qa_data.to_parquet('./qa.parquet', './corpus.parquet')
```

---

## 유틸리티 함수

### 최적 설정 추출

```python
from autorag.deploy import extract_best_config

extract_best_config(
    trial_path: str,             # 트리얼 경로
    output_path: str             # 출력 YAML 경로
)
```
