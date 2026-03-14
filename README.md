# 부동산 실거래가 분석 대시보드

서울·경기 아파트 실거래가와 거시경제 지표를 통합 분석하는 인터랙티브 대시보드

## 데이터 소스

- **국토교통부 실거래가 Open API** - 아파트 매매/전월세 실거래가
- **한국은행 ECOS API** - 기준금리, CPI, M2
- **yfinance** - 금, 유가, 원달러 환율

## 분석 기간

2020년 1월 ~ 2025년 12월

## 대상 지역

- 서울특별시 25개 자치구 전체
- 경기도 주요 19개 시·군

## 의존성 관리 원칙

- 단일 소스: `pyproject.toml`
- `requirements.txt` 는 유지하지 않음
- 기본 설치: `uv sync`
- 고급 패키지까지 설치: `uv sync --extra advanced`

## 로컬 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/<your-username>/apartment-analysis.git
cd apartment-analysis

# 2. Python 3.11 준비 + 가상환경 생성
uv python install 3.11
uv venv --python 3.11

# 3. 기본 의존성 동기화
uv sync

# 고급 분석용 optional 패키지까지 설치하려면
# uv sync --extra advanced

# 4. API 키 설정
cp .env.example .env
# .env 파일을 열어 실제 API 키 입력

# 5. 데이터 수집
uv run python scripts/run_full_pipeline.py

# 6. 집계만 다시 만들기
uv run python scripts/build_summary.py

# 7. 대시보드 실행
uv run streamlit run streamlit_app.py
```

## API 키 발급

| 서비스 | 발급 URL |
|--------|----------|
| 국토부 실거래가 API | https://www.data.go.kr |
| 한국은행 ECOS API | https://ecos.bok.or.kr |

## 프로젝트 구조

```
apartment-analysis/
├── config/settings.py          # 전역 설정 (API, 지역코드, 상수)
├── pipelines/                  # 데이터 수집 파이프라인
├── analysis/                   # 분석 모듈
├── dashboard/                  # Streamlit 대시보드
├── scripts/                    # 실행 스크립트
├── data/raw/                   # Raw 데이터 (git 제외)
├── data/processed/             # 집계 데이터
└── tests/                      # 테스트
```

## 라이선스

MIT
