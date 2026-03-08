# 부동산 실거래가 분석 대시보드 프로젝트 계획서

> **작성 목적**: Cursor AI에게 수행시킬 전체 프로젝트 구현 명세서  
> **분석 기간**: 2020년 1월 1일 ~ 2025년 12월 31일  
> **배포 환경**: GitHub (공개) + Streamlit Cloud

---

## 1. 프로젝트 개요

### 1.1 목적
서울 및 경기도 아파트 실거래가(매매·전월세)와 거시경제 지표(금리, 물가, 환율, 원자재)를 통합하여 시계열 분석 및 시각화를 제공하는 인터랙티브 대시보드를 구축한다.

### 1.2 핵심 기능
- 국토부 실거래가 Raw Data 수집 및 법정동+월 단위 파일 저장 파이프라인
- 한국은행 ECOS·yfinance 거시지표 수집 파이프라인
- 월별 Aggregation Summary 생성 모듈
- Streamlit 기반 멀티탭 분석 대시보드
- GitHub Actions를 활용한 정기 데이터 갱신

---

## 2. 프로젝트 디렉토리 구조

```
real-estate-dashboard/
│
├── .github/
│   └── workflows/
│       └── update_data.yml          # 월별 자동 데이터 갱신 워크플로우
│
├── config/
│   └── settings.py                  # API 엔드포인트, 지역코드, 공통 상수
│
├── data/
│   ├── raw/
│   │   ├── molit/                   # 국토부 실거래가 Raw
│   │   │   ├── apt_trade/           # 아파트 매매
│   │   │   │   └── {법정동코드}_{YYYYMM}.parquet
│   │   │   ├── apt_rent/            # 아파트 전월세
│   │   │   │   └── {법정동코드}_{YYYYMM}.parquet
│   │   │   └── building_ledger/     # 건축물대장
│   │   │       └── {법정동코드}.parquet
│   │   ├── ecos/                    # 한국은행 ECOS Raw
│   │   │   ├── bok_rate.parquet     # 한국 기준금리
│   │   │   ├── fed_rate.parquet     # 미국 기준금리
│   │   │   ├── cpi_kr.parquet       # 한국 CPI
│   │   │   ├── cpi_us.parquet       # 미국 CPI
│   │   │   └── m2.parquet           # M2 계절조정 말잔액
│   │   └── market/                  # yfinance Raw
│   │       ├── gold.parquet         # 금 가격 (GC=F)
│   │       ├── oil.parquet          # 국제유가 WTI (CL=F)
│   │       └── usdkrw.parquet       # 원달러 환율 (KRW=X)
│   │
│   └── processed/
│       ├── monthly_trade_summary.parquet    # 매매 월별 집계
│       ├── monthly_rent_summary.parquet     # 전월세 월별 집계
│       └── macro_monthly.parquet            # 거시지표 월별 통합
│
├── pipelines/
│   ├── __init__.py
│   ├── molit_pipeline.py            # 국토부 API 수집 파이프라인
│   ├── ecos_pipeline.py             # 한국은행 ECOS 수집 파이프라인
│   ├── market_pipeline.py           # yfinance 수집 파이프라인
│   └── aggregation_pipeline.py      # 월별 집계 파이프라인
│
├── analysis/
│   ├── __init__.py
│   ├── correlation.py               # 상관관계 분석
│   ├── trend.py                     # 트렌드/이동평균 분석
│   └── regional.py                  # 지역별 비교 분석
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py                       # Streamlit 메인 앱 진입점
│   └── pages/
│       ├── 01_overview.py           # 종합 현황 탭
│       ├── 02_trade_price.py        # 매매가 분석 탭
│       ├── 03_rent_price.py         # 전월세 분석 탭
│       ├── 04_macro_indicators.py   # 거시지표 탭
│       └── 05_correlation.py        # 복합 상관관계 탭
│
├── scripts/
│   ├── run_full_pipeline.py         # 전체 파이프라인 일괄 실행
│   └── build_summary.py             # 집계 파일만 재생성
│
├── tests/
│   ├── test_molit_pipeline.py
│   ├── test_ecos_pipeline.py
│   └── test_aggregation.py
│
├── .env.example                     # API 키 예시 (실제 키 미포함)
├── .gitignore                       # .env, data/raw/ 등 제외
├── requirements.txt
├── README.md
└── streamlit_app.py                 # Streamlit Cloud 진입점 (app.py 호출)
```

---

## 3. 데이터 소스 명세

### 3.1 국토부 실거래가 Open API

| 항목 | 내용 |
|------|------|
| API | 국토교통부 실거래가 공개시스템 |
| 엔드포인트 | `http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev` (매매) / `RTMSDataSvcAptRent` (전월세) |
| 인증 | Service Key (공공데이터포털 발급) |
| 수집 단위 | 법정동코드(5자리) × 월(YYYYMM) |
| 대상 지역 | 서울특별시 전체 자치구 + 경기도 주요 시군 |
| 수집 컬럼 (매매) | 법정동, 아파트명, 전용면적, 층, 건축연도, 거래금액, 거래연월일, 도로명주소 |
| 수집 컬럼 (전월세) | 법정동, 아파트명, 전용면적, 층, 건축연도, 보증금액, 월세금액, 계약연월일, 전월세구분 |

#### 법정동 코드 목록 (config/settings.py에 정의)
- **서울**: 종로구(11010), 중구(11020), 용산구(11030), 성동구(11040), 광진구(11050), 동대문구(11060), 중랑구(11070), 성북구(11080), 강북구(11090), 도봉구(11100), 노원구(11110), 은평구(11120), 서대문구(11130), 마포구(11140), 양천구(11150), 강서구(11160), 구로구(11170), 금천구(11180), 영등포구(11190), 동작구(11200), 관악구(11210), 서초구(11220), 강남구(11230), 송파구(11240), 강동구(11250)
- **경기**: 수원시(41011~41015), 성남시(41131~41133), 용인시(41461~41463), 고양시(41281~41285), 부천시(41190), 안양시(41171~41172), 화성시(41590), 광명시(41210), 평택시(41220), 의왕시(41430), 군포시(41410), 안산시(41271~41273), 광주시(41610), 하남시(41450), 남양주시(41360), 구리시(41310), 의정부시(41150), 파주시(41480), 김포시(41570)

### 3.2 건축물대장 API

| 항목 | 내용 |
|------|------|
| API | 국토교통부 건축데이터개방 |
| 엔드포인트 | `http://apis.data.go.kr/1613000/ArchPmsService_v2` |
| 수집 컬럼 | 건물명, 법정동코드, 지번, 주용도, 건축면적, 연면적, 층수, 사용승인일, 세대수 |
| 수집 조건 | 주용도 = 아파트(02), 연립주택(03) |

### 3.3 한국은행 ECOS API

| 지표명 | 통계표코드 | 주기 | 비고 |
|--------|-----------|------|------|
| 한국 기준금리 | 722Y001 / 0101000 | 월 | % |
| 미국 기준금리 | 902Y007 / 0000001 | 월 | Fed Funds Rate |
| 한국 소비자물가지수 | 901Y009 / 0 | 월 | 2020=100 기준 |
| 미국 소비자물가지수 | 902Y003 / 0000001 | 월 | YoY % |
| M2 계절조정 말잔액 | 101Y002 / BBHS00 | 월 | 억원 |

- API 엔드포인트: `https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/100/{통계표코드}/MM/{시작연월}/{종료연월}/{항목코드}`

### 3.4 yfinance 시장 데이터

| 지표 | Ticker | 비고 |
|------|--------|------|
| 금 선물 가격 | `GC=F` | USD/oz, 월말 종가 |
| WTI 원유 선물 | `CL=F` | USD/배럴, 월말 종가 |
| 원달러 환율 | `KRW=X` | 원/달러, 월말 종가 |

- 수집 주기: 일별 → 월말 종가(resample 'M') 저장

---

## 4. 파이프라인 상세 명세

### 4.1 `pipelines/molit_pipeline.py`

```
[MolitPipeline 클래스]

__init__(self, api_key: str, save_dir: str)
  - api_key: 공공데이터포털 서비스 키
  - save_dir: data/raw/molit/ 경로

fetch_apt_trade(self, lawd_cd: str, deal_ymd: str) -> pd.DataFrame
  - lawd_cd: 법정동코드 5자리
  - deal_ymd: YYYYMM 형식
  - 반환: 해당 지역-월 매매 거래 DataFrame
  - 저장: data/raw/molit/apt_trade/{lawd_cd}_{deal_ymd}.parquet

fetch_apt_rent(self, lawd_cd: str, deal_ymd: str) -> pd.DataFrame
  - 전월세 거래 수집
  - 저장: data/raw/molit/apt_rent/{lawd_cd}_{deal_ymd}.parquet

run_full_collection(self, start_ym: str, end_ym: str, region_codes: list)
  - 전체 기간 × 전체 지역 루프 실행
  - 이미 존재하는 파일은 skip (증분 수집)
  - API 호출 간 time.sleep(0.2) 적용 (서버 부하 방지)
  - 오류 발생 시 failed_list.json에 기록 후 계속 진행
  - 진행 상황 tqdm 프로그레스바 출력

[파일 명명 규칙]
  - apt_trade/11230_202301.parquet  (강남구 2023년 1월)
  - apt_rent/11230_202301.parquet
```

### 4.2 `pipelines/ecos_pipeline.py`

```
[EcosPipeline 클래스]

__init__(self, api_key: str, save_dir: str)

fetch_statistic(self, stat_code: str, item_code: str, 
                start_ym: str, end_ym: str, label: str) -> pd.DataFrame
  - ECOS API 단일 지표 수집
  - 반환: date(datetime), value(float) 컬럼 DataFrame

run_all(self, start_ym: str, end_ym: str)
  - 5개 지표 순차 수집
  - 각각 data/raw/ecos/{label}.parquet 저장
  - 저장 컬럼: date, value, unit, stat_name
```

### 4.3 `pipelines/market_pipeline.py`

```
[MarketPipeline 클래스]

__init__(self, save_dir: str)

fetch_yfinance(self, ticker: str, label: str, 
               start: str, end: str) -> pd.DataFrame
  - yfinance.download() 호출
  - 월말 종가(resample 'M').last() 추출
  - 저장: data/raw/market/{label}.parquet

run_all(self, start: str, end: str)
  - gold, oil, usdkrw 순차 수집
  - 저장 컬럼: date, close, ticker
```

### 4.4 `pipelines/aggregation_pipeline.py`

```
[AggregationPipeline 클래스]

build_monthly_trade_summary() -> pd.DataFrame
  - data/raw/molit/apt_trade/ 전체 파일 로드
  - 월별 × 자치구별 집계
  - 집계 항목:
    * 거래건수
    * 평균 거래금액 (전용면적별: 60㎡이하, 60~85㎡, 85㎡초과)
    * 중앙값 거래금액
    * 거래금액 상위/하위 10% 절사평균
    * 평균 전용면적
    * 평균 건축연도 → 평균 건물 연령
  - 저장: data/processed/monthly_trade_summary.parquet

build_monthly_rent_summary() -> pd.DataFrame
  - 전월세 집계 (전세/월세 구분 포함)
  - 집계 항목: 거래건수, 평균보증금, 평균월세, 전세가율
  - 저장: data/processed/monthly_rent_summary.parquet

build_macro_monthly() -> pd.DataFrame
  - ecos/*.parquet + market/*.parquet 병합
  - date 기준 outer join → 월별 통합 테이블
  - 저장: data/processed/macro_monthly.parquet

run_all()
  - 위 3개 함수 순차 실행
```

---

## 5. Streamlit 대시보드 명세

### 5.1 `streamlit_app.py` (진입점)
- `st.set_page_config(layout="wide")` 설정
- 사이드바: 지역 선택, 기간 범위 슬라이더, 면적 구분 필터
- `dashboard/pages/` 하위 멀티페이지 구성

### 5.2 Page 01 – 종합 현황 (`01_overview.py`)
- **KPI 카드 (4개)**: 최근 월 평균 매매가, 전월세 비율, 거래량 전월비, 기준금리
- **라인차트**: 서울 전체 월별 평균 매매가 추이 (2020~현재)
- **히트맵**: 자치구별 × 연도별 평균 거래금액 (plotly heatmap)
- **지도 시각화**: 자치구별 평균 가격 choropleth (folium 또는 pydeck)

### 5.3 Page 02 – 매매가 분석 (`02_trade_price.py`)
- 좌측: 지역 × 면적대별 매매가 시계열 (멀티라인)
- 우측: 거래량 바차트 (월별)
- 전용면적 구간별 가격 분포 박스플롯 (연도별 비교)
- YoY 상승률 라인 + 구간 음영 (코로나/금리인상 구간 강조)

### 5.4 Page 03 – 전월세 분석 (`03_rent_price.py`)
- 전세 / 월세 탭 분리
- 전세가율 (전세보증금 / 매매가) 추이
- 월세 전환율 계산 시각화
- 보증금 × 월세 산점도 (면적대별 색상 구분)

### 5.5 Page 04 – 거시지표 (`04_macro_indicators.py`)
- 이중 Y축 차트: 아파트 매매가 지수 vs 기준금리(한국/미국)
- 서브플롯 (2×3): CPI(한/미), M2, 금가격, 유가, 환율 시계열
- 지표별 전월비/전년비 증감률 테이블

### 5.6 Page 05 – 복합 상관관계 (`05_correlation.py`)
- 상관계수 히트맵 (매매가, 전세가, 금리, CPI, M2, 금, 유가, 환율)
- 시차(lag) 상관분석: lag 0~12개월 슬라이더로 조작
- 산점도 매트릭스 (seaborn pairplot → plotly 대체)
- 주요 지표 간 회귀선 + R² 표시

---

## 6. 보안 및 환경 설정

### 6.1 `.env` 파일 구조
```env
# .env (절대 git에 업로드 금지)
MOLIT_API_KEY=your_molit_service_key_here
ECOS_API_KEY=your_ecos_api_key_here
```

### 6.2 `.env.example` (git에 포함)
```env
# API 키 설정 방법:
# 1. 이 파일을 복사하여 .env로 저장
# 2. 각 값에 실제 발급받은 API 키 입력
MOLIT_API_KEY=
ECOS_API_KEY=
```

### 6.3 `.gitignore` 필수 항목
```gitignore
# 환경변수 / 시크릿
.env
*.env
secrets.toml

# Raw 데이터 (용량 이슈 + 재현 가능)
data/raw/

# Processed 데이터 (재생성 가능)
data/processed/

# Python 캐시
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# 가상환경
venv/
.venv/
env/
```

### 6.4 Streamlit Cloud 시크릿 설정
- Streamlit Cloud 대시보드 → App Settings → Secrets 메뉴에서 `.env` 내용을 `st.secrets` 형식으로 등록
```toml
# streamlit secrets (Streamlit Cloud 설정 화면에 직접 입력)
MOLIT_API_KEY = "your_key"
ECOS_API_KEY = "your_key"
```
- `config/settings.py`에서 키 로딩 로직:
```python
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # 로컬 개발 환경

def get_api_key(name: str) -> str:
    # Streamlit Cloud 우선, 없으면 환경변수
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, "")
```

---

## 7. `requirements.txt`

```
# 데이터 수집
requests==2.31.0
yfinance==0.2.36
python-dotenv==1.0.0

# 데이터 처리
pandas==2.1.4
numpy==1.26.4
pyarrow==15.0.0   # parquet I/O

# 시각화
plotly==5.18.0
folium==0.16.0
streamlit-folium==0.18.0

# 대시보드
streamlit==1.32.0

# 유틸리티
tqdm==4.66.1
loguru==0.7.2

# 테스트
pytest==8.0.0
```

---

## 8. 구현 순서 (Cursor 수행 단계)

### Phase 1 – 프로젝트 기반 설정
1. 디렉토리 구조 전체 생성 (`mkdir -p` 명령 일괄 실행)
2. `requirements.txt`, `.gitignore`, `.env.example`, `README.md` 작성
3. `config/settings.py` 작성 (법정동 코드 딕셔너리, API 엔드포인트 상수, 기간 상수 정의)
4. Git 초기화 및 GitHub 원격 저장소 연결

### Phase 2 – 데이터 수집 파이프라인
5. `pipelines/molit_pipeline.py` 구현 및 단위 테스트
   - 강남구 2023년 1월 샘플 수집으로 검증
6. `pipelines/ecos_pipeline.py` 구현 및 단위 테스트
7. `pipelines/market_pipeline.py` 구현 및 단위 테스트
8. `scripts/run_full_pipeline.py` 작성 (전체 기간 배치 실행)
9. 전체 Raw Data 수집 실행 (2020-01 ~ 2025-12)

### Phase 3 – 집계 파이프라인
10. `pipelines/aggregation_pipeline.py` 구현
11. `scripts/build_summary.py` 작성
12. 집계 파일 생성 검증 (컬럼 확인, 결측치 점검)

### Phase 4 – 분석 모듈
13. `analysis/trend.py`: 이동평균, YoY 계산 함수
14. `analysis/correlation.py`: lag 상관분석 함수
15. `analysis/regional.py`: 지역별 비교 함수

### Phase 5 – Streamlit 대시보드
16. `dashboard/app.py` 메인 레이아웃 및 사이드바 구현
17. Page 01 (종합 현황) 구현
18. Page 02 (매매가 분석) 구현
19. Page 03 (전월세 분석) 구현
20. Page 04 (거시지표) 구현
21. Page 05 (복합 상관관계) 구현
22. `streamlit_app.py` 진입점 연결

### Phase 6 – 배포 및 자동화
23. `.github/workflows/update_data.yml` 작성 (매월 1일 자동 수집)
24. Streamlit Cloud 배포 설정 (GitHub 연동, Secrets 등록)
25. README.md 최종 작성 (배지, 스크린샷, 설치 가이드 포함)

---

## 9. GitHub Actions 자동화 워크플로우

```yaml
# .github/workflows/update_data.yml
name: Monthly Data Update

on:
  schedule:
    - cron: '0 1 1 * *'   # 매월 1일 01:00 UTC (한국 10:00)
  workflow_dispatch:        # 수동 실행 허용

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run data pipeline
        env:
          MOLIT_API_KEY: ${{ secrets.MOLIT_API_KEY }}
          ECOS_API_KEY: ${{ secrets.ECOS_API_KEY }}
        run: python scripts/run_full_pipeline.py --mode incremental

      - name: Build summary
        run: python scripts/build_summary.py

      - name: Commit and push processed data
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/processed/
          git diff --staged --quiet || git commit -m "chore: monthly data update $(date +'%Y-%m')"
          git push
```

> **주의**: `data/raw/`는 용량 문제로 git에 포함하지 않고, `data/processed/`만 커밋. Raw 데이터는 로컬 또는 별도 스토리지(S3 등) 관리 권장.

---

## 10. 주요 설계 원칙 및 주의사항

### 데이터 수집
- **증분 수집**: 이미 저장된 `{lawd_cd}_{YYYYMM}.parquet` 파일은 skip하여 불필요한 API 재호출 방지
- **오류 내성**: 개별 API 호출 실패 시 `failed_list.json`에 기록 후 다음 항목 계속 수집
- **API 속도 제한**: 국토부 API 호출 간 `time.sleep(0.2)` 준수
- **Parquet 포맷**: CSV 대비 파일 크기 70% 절감, 컬럼 타입 보존

### 집계 파이프라인
- Raw 파일 전체를 메모리에 올리지 않고 `pd.read_parquet()` + 배치 처리
- 면적 구간 컬럼을 집계 시 파생: `pd.cut(area, bins=[0,60,85,np.inf])`
- 이상치 처리: 거래금액 상하위 1% 제거 후 통계 계산

### 대시보드
- 데이터 로딩에 `@st.cache_data(ttl=3600)` 데코레이터 적용
- 사이드바 필터 변경 시만 리렌더링되도록 최적화
- 모바일 대응을 위해 `use_container_width=True` 차트 기본 설정

### 보안
- `.env`는 절대 git에 포함 금지 (`.gitignore`로 차단)
- `st.secrets`를 통한 키 주입 (Streamlit Cloud 배포 시)
- API 키가 로그/에러 메시지에 노출되지 않도록 `loguru` 로거에서 마스킹 처리

---

## 11. README.md 구성 (GitHub 공개용)

```markdown
# 🏠 부동산 실거래가 분석 대시보드

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

서울·경기 아파트 실거래가와 거시경제 지표를 통합 분석하는 인터랙티브 대시보드

## 📊 데이터 소스
- 국토교통부 실거래가 Open API
- 한국은행 ECOS API
- yfinance (금, 유가, 환율)

## 🚀 로컬 실행 방법
1. 저장소 클론
2. `pip install -r requirements.txt`
3. `.env.example`을 `.env`로 복사 후 API 키 입력
4. `python scripts/run_full_pipeline.py` (첫 실행 시 전체 수집)
5. `streamlit run streamlit_app.py`

## 🔑 API 키 발급
- 국토부 API: https://www.data.go.kr
- 한국은행 ECOS: https://ecos.bok.or.kr
```

---

*본 계획서는 Cursor AI에게 단계별로 지시하여 구현하도록 설계되었습니다. Phase 1부터 순서대로 진행하며, 각 Phase 완료 후 테스트를 검증한 뒤 다음 단계로 진행하세요.*
