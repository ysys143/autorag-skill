# AutoRAG 설정 파일 레퍼런스

## 기본 구조

```yaml
# 벡터DB 설정 (선택)
vectordb:
  - name: default
    db_type: chroma
    embedding_model: openai

# 노드 라인 정의 (필수)
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: semantic_retrieval
        strategy:
          metrics: [...]
        modules:
          - module_type: vectordb
            ...
```

---

## vectordb 설정

```yaml
vectordb:
  - name: default                    # 참조용 이름
    db_type: chroma                  # 벡터DB 타입
    client_type: persistent          # persistent 또는 ephemeral
    embedding_model: openai          # 임베딩 모델
    collection_name: my_collection   # 컬렉션 이름
    path: ${PROJECT_DIR}/data/chroma # 저장 경로
```

**지원 벡터DB**

| db_type | 설명 |
|---------|------|
| `chroma` | Chroma DB |
| `milvus` | Milvus |
| `weaviate` | Weaviate |
| `pinecone` | Pinecone |
| `qdrant` | Qdrant |
| `couchbase` | Couchbase |

**지원 임베딩 모델**

| embedding_model | 설명 |
|-----------------|------|
| `openai` | OpenAI Embeddings |
| `openai_like` | OpenAI 호환 API |
| `huggingface` | HuggingFace 모델 |
| `ollama` | Ollama 임베딩 |

---

## node_lines 구조

```yaml
node_lines:
  - node_line_name: retrieve_node_line  # 노드 라인 이름
    nodes:
      - node_type: semantic_retrieval   # 노드 타입
        strategy:                        # 평가 전략
          metrics: [retrieval_f1]
        top_k: 3                         # 노드 레벨 파라미터
        modules:                         # 평가할 모듈 목록
          - module_type: vectordb
            vectordb: default
```

---

## 노드 타입별 설정

### query_expansion (쿼리 확장)

```yaml
- node_type: query_expansion
  strategy:
    metrics: [retrieval_f1, retrieval_recall]
    retrieval_modules:
      - module_type: bm25
      - module_type: vectordb
        vectordb: default
  modules:
    - module_type: pass_query_expansion    # 확장 없음
    - module_type: query_decompose         # 쿼리 분해
      generator_module_type: llama_index_llm
      llm: openai
      model: [gpt-4o-mini]
    - module_type: hyde                    # HyDE
      generator_module_type: llama_index_llm
      llm: openai
      model: [gpt-4o-mini]
    - module_type: multi_query_expansion   # 다중 쿼리
      generator_module_type: llama_index_llm
      llm: openai
      temperature: [0.2, 1.0]
```

### lexical_retrieval (BM25 검색)

```yaml
- node_type: lexical_retrieval
  strategy:
    metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
  top_k: 10
  modules:
    - module_type: bm25
      bm25_tokenizer: porter_stemmer  # 영어: porter_stemmer, space, gpt2
                                       # 한국어: kiwi, okt, kkma
```

### semantic_retrieval (벡터 검색)

```yaml
- node_type: semantic_retrieval
  strategy:
    metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
  top_k: 10
  modules:
    - module_type: vectordb
      vectordb: default
```

### hybrid_retrieval (하이브리드 검색)

```yaml
- node_type: hybrid_retrieval
  strategy:
    metrics: [retrieval_f1, retrieval_recall]
  top_k: 10
  modules:
    - module_type: hybrid_rrf           # RRF 기반
      weight_range: (4, 80)
    - module_type: hybrid_cc            # CC 기반
      normalize_method: [mm, tmm, z, dbsf]
      weight_range: (0.0, 1.0)
      test_weight_size: 101
```

### passage_reranker (재순위)

```yaml
- node_type: passage_reranker
  strategy:
    metrics: [retrieval_f1, retrieval_recall]
  top_k: 5
  modules:
    - module_type: pass_reranker        # 재순위 없음
    - module_type: monot5               # MonoT5
    - module_type: tart                 # TART
    - module_type: rankgpt              # RankGPT
      llm: openai
      model: gpt-4o-mini
    - module_type: cohere_reranker      # Cohere
    - module_type: flag_embedding_reranker  # FlagEmbedding
```

### passage_filter (필터링)

```yaml
- node_type: passage_filter
  strategy:
    metrics: [retrieval_f1, retrieval_recall]
  modules:
    - module_type: pass_passage_filter           # 필터 없음
    - module_type: similarity_threshold_cutoff   # 유사도 임계값
      threshold: 0.85
    - module_type: similarity_percentile_cutoff  # 유사도 백분위
      percentile: 0.6
    - module_type: threshold_cutoff              # 점수 임계값
      threshold: 0.85
    - module_type: percentile_cutoff             # 점수 백분위
      percentile: 0.6
    - module_type: recency_filter                # 날짜 필터
      threshold_datetime: 2015-01-01 3:45:07
```

### passage_compressor (압축)

```yaml
- node_type: passage_compressor
  strategy:
    metrics: [retrieval_token_f1, retrieval_token_recall]
  modules:
    - module_type: pass_compressor      # 압축 없음
    - module_type: tree_summarize       # 트리 요약
      llm: openai
      model: gpt-4o-mini
    - module_type: refine               # 반복 정제
      llm: openai
      model: gpt-4o-mini
```

### prompt_maker (프롬프트 구성)

```yaml
- node_type: prompt_maker
  strategy:
    metrics: [bleu, meteor, rouge]
    generator_modules:                  # 평가용 생성 모듈
      - module_type: llama_index_llm
        llm: openai
        model: [gpt-4o-mini]
  modules:
    - module_type: fstring
      prompt: "Question: {query}\nContext: {retrieved_contents}\nAnswer:"
    - module_type: long_context_reorder
      prompt: "Question: {query}\nContext: {retrieved_contents}\nAnswer:"
    - module_type: window_replacement
      prompt: "Question: {query}\nContext: {retrieved_contents}\nAnswer:"
```

**프롬프트 변수**
- `{query}`: 사용자 질문
- `{retrieved_contents}`: 검색된 문서 내용
- `{retrieved_ids}`: 검색된 문서 ID

### generator (답변 생성)

```yaml
- node_type: generator
  strategy:
    metrics:
      - metric_name: bleu
      - metric_name: meteor
      - metric_name: rouge
      - metric_name: sem_score
        embedding_model: openai
      - metric_name: g_eval
  modules:
    - module_type: llama_index_llm
      llm: openai
      model: [gpt-4o-mini, gpt-4o]
      temperature: [0.5, 1.0]
    - module_type: openai_llm
      llm: gpt-4o-mini
      temperature: 0.8
```

**LLM 타입**

| llm 값 | 설명 |
|--------|------|
| `openai` | OpenAI API |
| `openai_like` | OpenAI 호환 API |
| `ollama` | Ollama 로컬 |
| `huggingface` | HuggingFace 모델 |
| `bedrock` | AWS Bedrock |

---

## 전략(strategy) 설정

```yaml
strategy:
  metrics: [retrieval_f1, retrieval_recall]  # 평가 지표
  speed_threshold: 10                         # 속도 제한 (초)
  top_k: 5                                    # 검색 결과 수

  # prompt_maker에서 사용
  generator_modules:
    - module_type: llama_index_llm
      llm: openai
      model: gpt-4o-mini

  # query_expansion에서 사용
  retrieval_modules:
    - module_type: bm25
    - module_type: vectordb
      vectordb: default
```

---

## 평가 지표

### 검색 지표

| 지표 | 설명 |
|-----|------|
| `retrieval_f1` | F1 스코어 |
| `retrieval_recall` | 재현율 |
| `retrieval_precision` | 정밀도 |
| `retrieval_ndcg` | NDCG |
| `retrieval_mrr` | MRR |
| `retrieval_map` | MAP |

### 생성 지표

| 지표 | 설명 | 추가 옵션 |
|-----|------|----------|
| `bleu` | BLEU 스코어 | - |
| `meteor` | METEOR 스코어 | - |
| `rouge` | ROUGE 스코어 | - |
| `sem_score` | 의미적 유사도 | `embedding_model` |
| `bert_score` | BERTScore | `lang` (ko, en 등) |
| `g_eval` | LLM 기반 평가 | - |

---

## 환경 변수 사용

설정 파일에서 `${ENV_VAR}` 문법으로 환경 변수를 참조합니다.

```yaml
vectordb:
  - name: default
    path: ${PROJECT_DIR}/data/chroma

node_lines:
  - nodes:
      - modules:
          - module_type: llama_index_llm
            api_key: ${OPENAI_API_KEY}
```

---

## 파라미터 조합 자동화

리스트로 지정하면 모든 조합이 자동으로 평가됩니다.

```yaml
modules:
  - module_type: llama_index_llm
    llm: openai
    model: [gpt-4o-mini, gpt-3.5-turbo]  # 2개 모델
    temperature: [0.5, 1.0, 1.5]          # 3개 온도
    # 총 2 * 3 = 6개 조합 평가
```
