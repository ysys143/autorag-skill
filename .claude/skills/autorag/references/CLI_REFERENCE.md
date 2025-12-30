# AutoRAG CLI 명령어 레퍼런스

## evaluate

RAG 파이프라인 최적화를 실행합니다.

```bash
autorag evaluate \
    --config config.yaml \
    --qa_data_path qa.parquet \
    --corpus_data_path corpus.parquet \
    [--project_dir ./project] \
    [--skip_validation false]
```

| 옵션 | 필수 | 설명 |
|-----|------|------|
| `--config, -c` | O | 설정 YAML 파일 경로 |
| `--qa_data_path` | O | QA 데이터 Parquet 파일 경로 |
| `--corpus_data_path` | O | Corpus 데이터 Parquet 파일 경로 |
| `--project_dir` | X | 결과 저장 디렉토리 (기본: 현재 디렉토리) |
| `--skip_validation` | X | 사전 검증 건너뛰기 (기본: false) |

**예제**
```bash
# 기본 실행
autorag evaluate -c config.yaml --qa_data_path qa.parquet --corpus_data_path corpus.parquet

# 프로젝트 디렉토리 지정
autorag evaluate -c config.yaml --qa_data_path qa.parquet --corpus_data_path corpus.parquet --project_dir ./my_project

# 검증 건너뛰기 (빠른 프로토타이핑)
autorag evaluate -c config.yaml --qa_data_path qa.parquet --corpus_data_path corpus.parquet --skip_validation true
```

---

## validate

설정 파일과 데이터를 검증합니다. 본격적인 최적화 전에 실행하여 오류를 사전에 발견합니다.

```bash
autorag validate \
    --config config.yaml \
    --qa_data_path qa.parquet \
    --corpus_data_path corpus.parquet
```

| 옵션 | 필수 | 설명 |
|-----|------|------|
| `--config, -c` | O | 설정 YAML 파일 경로 |
| `--qa_data_path` | O | QA 데이터 Parquet 파일 경로 |
| `--corpus_data_path` | O | Corpus 데이터 Parquet 파일 경로 |

---

## run_api

최적화된 RAG 파이프라인을 REST API 서버로 실행합니다.

```bash
autorag run_api \
    --trial_dir ./project/0 \
    [--config_path best_config.yaml] \
    [--project_dir ./project] \
    [--host 0.0.0.0] \
    [--port 8000] \
    [--remote false]
```

| 옵션 | 설명 |
|-----|------|
| `--trial_dir` | 트리얼 디렉토리 경로 |
| `--config_path` | 추출된 설정 YAML 경로 (trial_dir 대신 사용) |
| `--project_dir` | 프로젝트 디렉토리 (config_path와 함께 사용) |
| `--host` | 호스트 주소 (기본: 0.0.0.0) |
| `--port` | 포트 번호 (기본: 8000) |
| `--remote` | ngrok 터널 생성 (기본: false) |

**예제**
```bash
# 트리얼 폴더에서 실행
autorag run_api --trial_dir ./project/0

# 포트 지정
autorag run_api --trial_dir ./project/0 --port 8080

# 원격 접속 (ngrok)
autorag run_api --trial_dir ./project/0 --remote true

# 추출된 설정으로 실행
autorag run_api --config_path best_config.yaml --project_dir ./project
```

**API 엔드포인트**

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/v1/run` | POST | 질문 실행 및 답변 반환 |
| `/v1/retrieve` | POST | 검색만 실행 |
| `/v1/stream` | POST | 스트리밍 응답 |
| `/version` | GET | 버전 정보 |

**요청 예시**
```bash
curl -X POST http://localhost:8000/v1/run \
    -H "Content-Type: application/json" \
    -d '{"query": "What is machine learning?"}'
```

---

## run_web

Streamlit 기반 웹 인터페이스를 실행합니다.

```bash
autorag run_web \
    --trial_path ./project/0
# 또는
autorag run_web \
    --yaml_path best_config.yaml \
    --project_dir ./project
```

| 옵션 | 설명 |
|-----|------|
| `--trial_path` | 트리얼 디렉토리 경로 |
| `--yaml_path` | 설정 YAML 경로 (trial_path 대신 사용) |
| `--project_dir` | 프로젝트 디렉토리 (yaml_path와 함께 사용) |

---

## dashboard

Panel 기반 대시보드를 실행하여 최적화 결과를 시각화합니다.

```bash
autorag dashboard \
    --trial_dir ./project/0 \
    [--port 7690]
```

| 옵션 | 필수 | 설명 |
|-----|------|------|
| `--trial_dir` | O | 트리얼 디렉토리 경로 |
| `--port` | X | 포트 번호 (기본: 7690) |

**예제**
```bash
autorag dashboard --trial_dir ./project/0
autorag dashboard --trial_dir ./project/0 --port 8888
```

---

## extract_best_config

최적화된 파이프라인을 별도 YAML 파일로 추출합니다.

```bash
autorag extract_best_config \
    --trial_path ./project/0 \
    --output_path best_config.yaml
```

| 옵션 | 필수 | 설명 |
|-----|------|------|
| `--trial_path` | O | 트리얼 디렉토리 경로 |
| `--output_path` | O | 출력 YAML 파일 경로 |

---

## restart_evaluate

중단된 최적화를 재개합니다.

```bash
autorag restart_evaluate --trial_path ./project/0
```

| 옵션 | 필수 | 설명 |
|-----|------|------|
| `--trial_path` | O | 중단된 트리얼 디렉토리 경로 |

**사용 시나리오**
- 네트워크 오류로 중단된 경우
- 시스템 리소스 부족으로 중단된 경우
- 수동으로 중단한 경우
