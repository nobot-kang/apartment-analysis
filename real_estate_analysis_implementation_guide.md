# 부동산 실거래가 분석 대시보드 — 분석 구현 상세 가이드

> **목적**: Cursor AI에게 20개 분석 아이디어를 단계별로 구현시키기 위한 상세 명세서  
> **패키지 관리**: `uv venv` 기반 가상환경  
> **전제 조건**: `real_estate_project_plan.md` 기반 프로젝트 구조 완성 후 적용

---

## 0. 개발 환경 설정 (uv venv)

### 0.1 uv 설치 및 가상환경 초기화

```bash
# uv 설치 (최초 1회)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 루트에서 가상환경 생성
uv venv .venv --python 3.11

# 가상환경 활성화
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# 기본 패키지 설치
uv pip install -r requirements.txt
```

### 0.2 분석별 추가 패키지 설치

각 Level에서 필요한 패키지는 아래 명령어로 추가 설치합니다.  
설치 후 `uv pip freeze > requirements.txt`로 고정합니다.

```bash
# Level 1~2 추가
uv pip install plotly folium streamlit-folium

# Level 3 추가
uv pip install scipy statsmodels

# Level 4 추가
uv pip install scikit-learn dtaidistance hmmlearn geopandas pyproj
```

### 0.3 `pyproject.toml` (uv 프로젝트 관리 시 권장)

```toml
[project]
name = "real-estate-dashboard"
version = "0.1.0"
requires-python = ">=3.11"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
]
```

---

## 🟢 Level 1 — 기초 현황 파악

---

### Analysis 01. 월별 거래량 추이 분석

#### 목표
서울·경기 전체 아파트 매매·전월세 거래량을 월별로 집계하여 계절성, 정책 이벤트, 거래 절벽 구간을 시각화한다.

#### 추가 패키지
```bash
# 별도 설치 없음 (pandas, plotly 기본)
```

#### 파일 위치
```
analysis/basic/01_monthly_volume.py
dashboard/pages/01_overview.py  ← 차트 컴포넌트 임포트
```

#### 구현 코드 명세

```python
# analysis/basic/01_monthly_volume.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

DATA_DIR = Path("data/processed")

def load_monthly_volume() -> pd.DataFrame:
    """
    monthly_trade_summary.parquet, monthly_rent_summary.parquet 로드 후
    거래유형(매매/전세/월세) 컬럼을 추가하여 long format으로 반환.
    반환 컬럼: ['year_month', 'region', 'deal_type', 'count']
    """
    trade = pd.read_parquet(DATA_DIR / "monthly_trade_summary.parquet")
    rent  = pd.read_parquet(DATA_DIR / "monthly_rent_summary.parquet")

    trade_long = trade[["year_month", "region", "trade_count"]].copy()
    trade_long["deal_type"] = "매매"
    trade_long = trade_long.rename(columns={"trade_count": "count"})

    jeonse = rent[rent["rent_type"] == "전세"][["year_month", "region", "rent_count"]].copy()
    jeonse["deal_type"] = "전세"
    jeonse = jeonse.rename(columns={"rent_count": "count"})

    wolse = rent[rent["rent_type"] == "월세"][["year_month", "region", "rent_count"]].copy()
    wolse["deal_type"] = "월세"
    wolse = wolse.rename(columns={"rent_count": "count"})

    return pd.concat([trade_long, jeonse, wolse], ignore_index=True)


def build_volume_chart(
    df: pd.DataFrame,
    region: str = "서울전체",
    highlight_events: bool = True
) -> go.Figure:
    """
    Parameters
    ----------
    df           : load_monthly_volume() 반환 DataFrame
    region       : 필터링할 지역명
    highlight_events : 주요 정책 이벤트 수직선 표시 여부

    Returns
    -------
    plotly Figure — 매매/전세/월세 3개 라인 + 거래량 바차트
    """
    filtered = df[df["region"] == region].copy()
    filtered["year_month"] = pd.to_datetime(filtered["year_month"].astype(str), format="%Y%m")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("거래유형별 월 거래량", "매매 거래량 바차트"),
        vertical_spacing=0.1
    )

    colors = {"매매": "#E63946", "전세": "#457B9D", "월세": "#2A9D8F"}
    for deal in ["매매", "전세", "월세"]:
        sub = filtered[filtered["deal_type"] == deal]
        fig.add_trace(
            go.Scatter(
                x=sub["year_month"], y=sub["count"],
                name=deal, line=dict(color=colors[deal], width=2),
                mode="lines+markers", marker=dict(size=4)
            ),
            row=1, col=1
        )

    trade_sub = filtered[filtered["deal_type"] == "매매"]
    fig.add_trace(
        go.Bar(
            x=trade_sub["year_month"], y=trade_sub["count"],
            name="매매(바)", marker_color="#E63946", opacity=0.6
        ),
        row=2, col=1
    )

    # 주요 이벤트 수직선
    if highlight_events:
        events = {
            "2020-06": "임대차3법",
            "2021-08": "가계대출규제",
            "2022-01": "금리인상시작",
            "2023-01": "특례보금자리론",
        }
        for date_str, label in events.items():
            fig.add_vline(
                x=pd.Timestamp(date_str),
                line_width=1, line_dash="dash", line_color="gray",
                annotation_text=label,
                annotation_position="top right"
            )

    fig.update_layout(
        height=600, hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        title_text=f"{region} 월별 아파트 거래량 추이 (2020~2025)"
    )
    return fig
```

#### Streamlit 연동 포인트
```python
# dashboard/pages/01_overview.py 내 해당 섹션
from analysis.basic.monthly_volume import load_monthly_volume, build_volume_chart

@st.cache_data(ttl=3600)
def get_volume_data():
    return load_monthly_volume()

df_vol = get_volume_data()
region = st.sidebar.selectbox("지역 선택", ["서울전체", "강남구", "노원구", ...])
st.plotly_chart(build_volume_chart(df_vol, region=region), use_container_width=True)
```

---

### Analysis 02. 자치구별 평균 매매가 랭킹

#### 목표
자치구별 평균 실거래가를 수평 바차트로 표현하고 연도 슬라이더로 순위 변동을 확인한다.

#### 추가 패키지
```bash
uv pip install plotly   # 이미 설치됨
```

#### 파일 위치
```
analysis/basic/02_district_ranking.py
```

#### 구현 코드 명세

```python
# analysis/basic/02_district_ranking.py

import pandas as pd
import plotly.express as px
from pathlib import Path

def load_district_avg(year: int) -> pd.DataFrame:
    """
    지정 연도의 자치구별 평균 매매가(만원/㎡) 반환.
    반환 컬럼: ['district', 'avg_price', 'avg_price_per_sqm', 'trade_count']
    """
    df = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    df["year"] = (df["year_month"] // 100)
    yearly = (
        df[df["year"] == year]
        .groupby("district")
        .agg(
            avg_price=("avg_price", "mean"),
            avg_price_per_sqm=("avg_price_per_sqm", "mean"),
            trade_count=("trade_count", "sum"),
        )
        .reset_index()
        .sort_values("avg_price", ascending=True)  # 수평 바차트용 오름차순
    )
    return yearly


def build_ranking_chart(df: pd.DataFrame, year: int, metric: str = "avg_price") -> px.Figure:
    """
    Parameters
    ----------
    df     : load_district_avg() 반환 DataFrame
    year   : 표시 연도 (차트 제목용)
    metric : 'avg_price' | 'avg_price_per_sqm'
    """
    label_map = {"avg_price": "평균 매매가 (만원)", "avg_price_per_sqm": "㎡당 평균가 (만원/㎡)"}
    fig = px.bar(
        df, x=metric, y="district",
        orientation="h",
        color=metric,
        color_continuous_scale="RdYlGn_r",
        text=df[metric].apply(lambda x: f"{x:,.0f}"),
        labels={metric: label_map[metric], "district": "자치구"},
        title=f"{year}년 자치구별 아파트 {label_map[metric]} 랭킹",
        hover_data={"trade_count": True}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=700, coloraxis_showscale=False)
    return fig


def build_ranking_animation(metric: str = "avg_price") -> px.Figure:
    """
    2020~2025년 전체 데이터를 사용하는 애니메이션 바차트.
    연도 슬라이더로 랭킹 변동 확인 가능.
    """
    df = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    df["year"] = (df["year_month"] // 100).astype(str)
    yearly = (
        df.groupby(["year", "district"])
        .agg(avg_price=("avg_price", "mean"), avg_price_per_sqm=("avg_price_per_sqm", "mean"))
        .reset_index()
    )
    fig = px.bar(
        yearly, x=metric, y="district",
        animation_frame="year",
        orientation="h",
        color=metric,
        color_continuous_scale="RdYlGn_r",
        range_x=[0, yearly[metric].max() * 1.15],
        title=f"자치구별 아파트 {metric} 랭킹 변화 (2020~2025)"
    )
    fig.update_layout(height=700)
    return fig
```

---

### Analysis 03. 전용면적 구간별 가격 분포 박스플롯

#### 목표
60㎡ 이하 / 60~85㎡ / 85㎡ 초과 구간별 연도별 가격 분포를 비교하여 면적에 따른 가격 패턴을 분석한다.

#### 추가 패키지
```bash
# 별도 설치 없음
```

#### 파일 위치
```
analysis/basic/03_area_price_dist.py
```

#### 구현 코드 명세

```python
# analysis/basic/03_area_price_dist.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

AREA_BINS   = [0, 60, 85, np.inf]
AREA_LABELS = ["소형(~60㎡)", "중형(60~85㎡)", "대형(85㎡~)"]

def load_raw_with_area_bin(district: str = None) -> pd.DataFrame:
    """
    apt_trade raw 파일을 모두 로드하고 면적 구간 컬럼 추가.
    district 지정 시 해당 자치구만 필터.
    반환 컬럼: ['year', 'area_bin', 'price', 'district']
    """
    import glob
    files = glob.glob("data/raw/molit/apt_trade/*.parquet")
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=["거래금액", "전용면적", "법정동", "년", "월"])
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    combined["price"] = (
        combined["거래금액"].str.replace(",", "").astype(float)
    )
    combined["area"]  = combined["전용면적"].astype(float)
    combined["year"]  = combined["년"].astype(str)
    combined["area_bin"] = pd.cut(
        combined["area"], bins=AREA_BINS, labels=AREA_LABELS, right=False
    )
    if district:
        combined = combined[combined["법정동"].str.startswith(district)]
    return combined[["year", "area_bin", "price", "법정동"]].dropna()


def build_boxplot(df: pd.DataFrame, area_bin: str) -> go.Figure:
    """
    선택한 면적 구간의 연도별 박스플롯.
    이상치(상하위 1%) 제거 후 표시.
    """
    sub = df[df["area_bin"] == area_bin].copy()
    # 이상치 1% 제거
    lo, hi = sub["price"].quantile(0.01), sub["price"].quantile(0.99)
    sub = sub[(sub["price"] >= lo) & (sub["price"] <= hi)]

    years = sorted(sub["year"].unique())
    fig = go.Figure()
    for yr in years:
        fig.add_trace(go.Box(
            y=sub[sub["year"] == yr]["price"],
            name=str(yr),
            boxmean="sd",
            marker_color=f"hsl({int(yr)*30 % 360}, 70%, 50%)"
        ))
    fig.update_layout(
        title=f"{area_bin} 연도별 매매가 분포",
        yaxis_title="거래금액 (만원)",
        xaxis_title="연도",
        height=500,
        showlegend=False
    )
    return fig
```

---

### Analysis 04. 건축 연령별 가격 프리미엄 분석

#### 목표
건축 연도 기준으로 신축/준신축/구축/노후 4구간으로 나눠 지역별 신축 프리미엄을 정량화한다.

#### 파일 위치
```
analysis/basic/04_age_premium.py
```

#### 구현 코드 명세

```python
# analysis/basic/04_age_premium.py

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

def classify_building_age(build_year: int, deal_year: int) -> str:
    age = deal_year - build_year
    if age <= 5:   return "신축(0~5년)"
    if age <= 15:  return "준신축(6~15년)"
    if age <= 30:  return "구축(16~30년)"
    return "노후(30년+)"

def load_age_premium_data() -> pd.DataFrame:
    """
    raw 파일에서 건축연도 + 거래금액 로드.
    건물 연령 구간 컬럼 추가.
    반환 컬럼: ['district', 'year', 'age_bin', 'price', 'price_per_sqm']
    """
    import glob
    files = glob.glob("data/raw/molit/apt_trade/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df["price"]         = df["거래금액"].str.replace(",", "").astype(float)
    df["area"]          = df["전용면적"].astype(float)
    df["price_per_sqm"] = df["price"] / df["area"]
    df["deal_year"]     = df["년"].astype(int)
    df["build_year"]    = df["건축년도"].astype(int)
    df["age_bin"]       = df.apply(
        lambda r: classify_building_age(r["build_year"], r["deal_year"]), axis=1
    )
    return df[["법정동", "deal_year", "age_bin", "price", "price_per_sqm"]].dropna()


def build_age_premium_chart(df: pd.DataFrame, district: str, year: int) -> px.Figure:
    sub = df[(df["법정동"].str.startswith(district)) & (df["deal_year"] == year)]
    agg = (
        sub.groupby("age_bin")["price_per_sqm"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    ORDER = ["신축(0~5년)", "준신축(6~15년)", "구축(16~30년)", "노후(30년+)"]
    agg["age_bin"] = pd.Categorical(agg["age_bin"], categories=ORDER, ordered=True)
    agg = agg.sort_values("age_bin")

    fig = px.bar(
        agg, x="age_bin", y="mean",
        error_y=agg["mean"] - agg["median"],
        color="mean",
        color_continuous_scale="Blues",
        labels={"mean": "평균 ㎡당 가격(만원)", "age_bin": "건물 연령"},
        title=f"{district} {year}년 건물 연령별 ㎡당 평균 매매가",
        text=agg["mean"].apply(lambda x: f"{x:,.0f}")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=450)
    return fig
```

---

### Analysis 05. 전세가율 추이 분석

#### 목표
매매가 대비 전세보증금 비율(전세가율)을 월별로 계산하여 갭투자 리스크 구간을 식별한다.

#### 파일 위치
```
analysis/basic/05_jeonse_ratio.py
```

#### 구현 코드 명세

```python
# analysis/basic/05_jeonse_ratio.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

RISK_THRESHOLD = 80.0  # 전세가율 80% 이상 = 리스크 구간

def load_jeonse_ratio() -> pd.DataFrame:
    """
    월별 집계 파일에서 전세가율 계산.
    같은 법정동 기준으로 매매 평균가 / 전세 평균보증금 병합.
    반환 컬럼: ['year_month', 'district', 'avg_trade', 'avg_jeonse', 'jeonse_ratio']
    """
    trade = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    rent  = pd.read_parquet(Path("data/processed/monthly_rent_summary.parquet"))

    jeonse = rent[rent["rent_type"] == "전세"].copy()
    merged = trade.merge(
        jeonse[["year_month", "district", "avg_deposit"]],
        on=["year_month", "district"],
        how="inner"
    )
    merged["jeonse_ratio"] = (merged["avg_deposit"] / merged["avg_price"] * 100).round(2)
    merged["year_month_dt"] = pd.to_datetime(merged["year_month"].astype(str), format="%Y%m")
    return merged


def build_jeonse_ratio_chart(df: pd.DataFrame, district: str) -> go.Figure:
    sub = df[df["district"] == district].sort_values("year_month_dt")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["year_month_dt"], y=sub["jeonse_ratio"],
        mode="lines+markers", name="전세가율(%)",
        line=dict(color="#2196F3", width=2)
    ))
    # 80% 기준선
    fig.add_hline(
        y=RISK_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="리스크 기준(80%)", annotation_position="bottom right"
    )
    # 리스크 구간 음영
    risk_mask = sub["jeonse_ratio"] >= RISK_THRESHOLD
    if risk_mask.any():
        fig.add_trace(go.Scatter(
            x=pd.concat([sub.loc[risk_mask, "year_month_dt"],
                         sub.loc[risk_mask, "year_month_dt"].iloc[::-1]]),
            y=pd.concat([sub.loc[risk_mask, "jeonse_ratio"],
                         pd.Series([RISK_THRESHOLD] * risk_mask.sum())]),
            fill="toself", fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="rgba(255,0,0,0)"),
            name="리스크 구간"
        ))
    fig.update_layout(
        title=f"{district} 아파트 전세가율 추이",
        yaxis_title="전세가율 (%)",
        height=450, hovermode="x unified"
    )
    return fig
```

---

## 🟡 Level 2 — 심화 비교 분석

---

### Analysis 06. 자치구별 가격 히트맵 (연도×지역)

#### 목표
행: 자치구, 열: 연도의 히트맵으로 지역간 가격 격차의 시간적 변화를 한눈에 파악한다.

#### 파일 위치
```
analysis/intermediate/06_heatmap.py
```

#### 구현 코드 명세

```python
# analysis/intermediate/06_heatmap.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def build_district_year_heatmap(metric: str = "avg_price") -> go.Figure:
    """
    Parameters
    ----------
    metric : 'avg_price' | 'avg_price_per_sqm' | 'yoy_change'
    """
    df = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    df["year"] = df["year_month"] // 100

    pivot = (
        df.groupby(["district", "year"])[metric]
        .mean()
        .unstack("year")
    )

    if metric == "yoy_change":
        # 전년 대비 변화율 계산
        pivot = pivot.pct_change(axis=1) * 100

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index,
        colorscale="RdYlGn_r" if metric != "yoy_change" else "RdBu_r",
        text=[[f"{v:,.1f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title=metric),
        zmid=0 if metric == "yoy_change" else None
    ))
    title_map = {
        "avg_price": "자치구별 연도별 평균 매매가 (만원)",
        "avg_price_per_sqm": "자치구별 연도별 ㎡당 평균가 (만원/㎡)",
        "yoy_change": "자치구별 연도별 YoY 가격 상승률 (%)"
    }
    fig.update_layout(
        title=title_map.get(metric, metric),
        xaxis_title="연도", yaxis_title="자치구",
        height=700
    )
    return fig
```

---

### Analysis 07. 층수별 가격 프리미엄 분석

#### 목표
동일 단지·면적 기준으로 저층/중층/고층 간 가격 차이를 지역별로 비교한다.

#### 파일 위치
```
analysis/intermediate/07_floor_premium.py
```

#### 구현 코드 명세

```python
# analysis/intermediate/07_floor_premium.py

import pandas as pd
import numpy as np
import plotly.express as px
import glob

FLOOR_BINS   = [0, 5, 15, np.inf]
FLOOR_LABELS = ["저층(1~5층)", "중층(6~15층)", "고층(16층+)"]

def load_floor_data(districts: list) -> pd.DataFrame:
    """
    법정동 코드 기준으로 해당 자치구 raw 파일만 로드.
    반환 컬럼: ['apt_name', 'area_bin', 'floor_bin', 'price_per_sqm', 'district']
    """
    pattern = "data/raw/molit/apt_trade/*.parquet"
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        code = f.split("/")[-1].split("_")[0]
        if any(code.startswith(d) for d in districts):
            dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    df["price"]         = df["거래금액"].str.replace(",", "").astype(float)
    df["area"]          = df["전용면적"].astype(float)
    df["floor"]         = df["층"].astype(float)
    df["price_per_sqm"] = df["price"] / df["area"]
    df["floor_bin"]     = pd.cut(df["floor"], bins=FLOOR_BINS, labels=FLOOR_LABELS, right=True)
    df["area_bin"]      = pd.cut(
        df["area"], bins=[0, 60, 85, np.inf],
        labels=["소형", "중형", "대형"], right=False
    )
    return df[["아파트", "area_bin", "floor_bin", "price_per_sqm", "법정동"]].dropna()


def build_floor_premium_chart(df: pd.DataFrame, compare_districts: list) -> px.Figure:
    agg = (
        df[df["법정동"].isin(compare_districts)]
        .groupby(["법정동", "floor_bin"])["price_per_sqm"]
        .mean().reset_index()
    )
    fig = px.bar(
        agg, x="floor_bin", y="price_per_sqm",
        color="법정동", barmode="group",
        labels={"price_per_sqm": "평균 ㎡당 가격(만원)", "floor_bin": "층 구간"},
        title="지역별 층수 구간별 ㎡당 평균 매매가 비교"
    )
    fig.update_layout(height=450)
    return fig
```

---

### Analysis 08. YoY 가격 상승률 지도 시각화

#### 목표
법정동별 전년 동월 대비 매매가 상승률을 choropleth 지도로 시각화한다.

#### 추가 패키지
```bash
uv pip install folium streamlit-folium geopandas requests
```

#### 파일 위치
```
analysis/intermediate/08_yoy_map.py
data/geojson/seoul_districts.geojson    ← 자치구 경계 GeoJSON (공공데이터 다운로드)
```

#### 구현 코드 명세

```python
# analysis/intermediate/08_yoy_map.py

import pandas as pd
import folium
import json
from pathlib import Path

GEOJSON_PATH = Path("data/geojson/seoul_districts.geojson")

def load_yoy_change(target_year: int) -> pd.DataFrame:
    """
    target_year와 target_year-1 의 자치구별 평균가 비교 → YoY 변화율 반환.
    반환 컬럼: ['district', 'avg_price_cur', 'avg_price_prev', 'yoy_pct']
    """
    df = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    df["year"] = df["year_month"] // 100
    yearly = df.groupby(["district", "year"])["avg_price"].mean().reset_index()
    cur  = yearly[yearly["year"] == target_year].rename(columns={"avg_price": "cur"})
    prev = yearly[yearly["year"] == target_year - 1].rename(columns={"avg_price": "prev"})
    merged = cur.merge(prev[["district", "prev"]], on="district", how="left")
    merged["yoy_pct"] = ((merged["cur"] - merged["prev"]) / merged["prev"] * 100).round(2)
    return merged


def build_choropleth_map(yoy_df: pd.DataFrame, target_year: int) -> folium.Map:
    """
    folium Choropleth 지도 생성.
    GeoJSON key: 'feature.properties.SIG_KOR_NM' (시군구 한글명)
    """
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        geo = json.load(f)

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=geo,
        name="YoY 가격 변화율",
        data=yoy_df,
        columns=["district", "yoy_pct"],
        key_on="feature.properties.SIG_KOR_NM",
        fill_color="RdYlGn",
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name=f"{target_year}년 YoY 아파트 매매가 변화율 (%)",
        nan_fill_color="lightgray"
    ).add_to(m)

    # 툴팁: 마우스 오버 시 자치구명 + 변화율 표시
    tooltip_df = yoy_df.set_index("district")["yoy_pct"].to_dict()
    for feature in geo["features"]:
        name = feature["properties"].get("SIG_KOR_NM", "")
        rate = tooltip_df.get(name, "N/A")
        folium.GeoJson(
            feature,
            tooltip=folium.GeoJsonTooltip(
                fields=["SIG_KOR_NM"],
                aliases=[f"{name}: {rate}%"]
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
```

#### GeoJSON 다운로드 스크립트
```python
# scripts/download_geojson.py
# 통계청 제공 행정구역 경계 GeoJSON 다운로드
import requests, json
from pathlib import Path

URL = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
Path("data/geojson").mkdir(parents=True, exist_ok=True)
resp = requests.get(URL, timeout=30)
with open("data/geojson/seoul_districts.geojson", "w", encoding="utf-8") as f:
    json.dump(resp.json(), f, ensure_ascii=False)
print("GeoJSON 저장 완료")
```

---

### Analysis 09. 거래량 vs 가격 선행-후행 관계 분석

#### 목표
거래량 감소가 가격 하락을 선행하는지 cross-correlation으로 검증한다.

#### 추가 패키지
```bash
uv pip install scipy
```

#### 파일 위치
```
analysis/intermediate/09_volume_price_lag.py
```

#### 구현 코드 명세

```python
# analysis/intermediate/09_volume_price_lag.py

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def compute_cross_correlation(
    series_x: pd.Series,
    series_y: pd.Series,
    max_lag: int = 12
) -> pd.DataFrame:
    """
    series_x 를 lag 이동 후 series_y 와의 pearson 상관계수 계산.
    lag > 0: series_x 가 series_y 를 선행
    반환 컬럼: ['lag', 'corr', 'pvalue']
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = series_x.shift(lag)
        mask = ~(shifted.isna() | series_y.isna())
        if mask.sum() < 10:
            continue
        corr, pvalue = pearsonr(shifted[mask], series_y[mask])
        results.append({"lag": lag, "corr": corr, "pvalue": pvalue})
    return pd.DataFrame(results)


def load_volume_price_series(district: str = "서울전체") -> pd.DataFrame:
    df = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    if district != "서울전체":
        df = df[df["district"] == district]
    monthly = df.groupby("year_month").agg(
        trade_count=("trade_count", "sum"),
        avg_price=("avg_price", "mean")
    ).reset_index().sort_values("year_month")
    monthly["price_mom"] = monthly["avg_price"].pct_change() * 100   # 전월비
    monthly["volume_mom"] = monthly["trade_count"].pct_change() * 100
    return monthly


def build_lag_correlation_chart(df: pd.DataFrame, district: str) -> go.Figure:
    lag_df = compute_cross_correlation(df["volume_mom"], df["price_mom"], max_lag=12)
    sig = lag_df["pvalue"] < 0.05

    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "거래량 vs 매매가 (전월비)", "Cross-Correlation (거래량→가격 선행 분석)"
    ))
    dt = pd.to_datetime(df["year_month"].astype(str), format="%Y%m")

    fig.add_trace(go.Scatter(x=dt, y=df["volume_mom"], name="거래량 전월비(%)",
                             line=dict(color="#2196F3")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dt, y=df["price_mom"], name="매매가 전월비(%)",
                             line=dict(color="#E63946")), row=1, col=1)

    colors = ["green" if s else "lightgray" for s in sig]
    fig.add_trace(go.Bar(
        x=lag_df["lag"], y=lag_df["corr"],
        marker_color=colors, name="상관계수"
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="black", row=2, col=1)
    fig.update_xaxes(title_text="Lag (개월, 양수=거래량이 가격 선행)", row=2, col=1)
    fig.update_layout(height=600, title_text=f"{district} 거래량-매매가 선행 분석")
    return fig
```

---

### Analysis 10. 전월세 전환율 역산 분석

#### 목표
혼합 계약(보증금+월세)에서 실질 전월세 전환율을 역산하고 법정 기준과의 괴리를 분석한다.

#### 파일 위치
```
analysis/intermediate/10_conversion_rate.py
```

#### 구현 코드 명세

```python
# analysis/intermediate/10_conversion_rate.py
# 전월세 전환율 = 월세 × 12 / (전세보증금 - 월세보증금) × 100

import pandas as pd
import plotly.graph_objects as go
import glob
from pathlib import Path

LEGAL_RATE = 5.5  # 법정 전월세 전환율 기준 (2024년 기준, settings.py에서 관리 권장)

def load_conversion_rate_data() -> pd.DataFrame:
    """
    월세 계약(보증금>0, 월세>0) 에서 전환율 역산.
    반환 컬럼: ['year_month', 'district', 'conversion_rate', 'sample_count']
    """
    files = glob.glob("data/raw/molit/apt_rent/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # 월세 계약만 (전월세구분 == '월세' 또는 보증금+월세 혼합)
    wolse = df[df["월세금액"].astype(float) > 0].copy()
    wolse["deposit"]  = wolse["보증금액"].str.replace(",","").astype(float)
    wolse["monthly"]  = wolse["월세금액"].str.replace(",","").astype(float)
    wolse["jeonse_equiv"] = wolse["deposit"] + (wolse["monthly"] * 12 / (LEGAL_RATE/100))

    # 전환율 = 월세 × 12 / (전세등가 - 실제보증금) 역산
    ref_jeonse = load_avg_jeonse_deposit()  # 같은 법정동 평균 전세보증금
    wolse = wolse.merge(ref_jeonse, on=["year_month","district"], how="left")
    wolse["conversion_rate"] = (
        (wolse["monthly"] * 12) / (wolse["ref_jeonse"] - wolse["deposit"])
    ) * 100
    wolse = wolse[wolse["conversion_rate"].between(1, 30)]  # 비현실값 제거

    monthly_agg = wolse.groupby(["year_month", "district"]).agg(
        conversion_rate=("conversion_rate", "median"),
        sample_count=("conversion_rate", "count")
    ).reset_index()
    return monthly_agg


def load_avg_jeonse_deposit() -> pd.DataFrame:
    df = pd.read_parquet(Path("data/processed/monthly_rent_summary.parquet"))
    jeonse = df[df["rent_type"] == "전세"][["year_month", "district", "avg_deposit"]].copy()
    return jeonse.rename(columns={"avg_deposit": "ref_jeonse"})


def build_conversion_rate_chart(df: pd.DataFrame, districts: list) -> go.Figure:
    fig = go.Figure()
    for d in districts:
        sub = df[df["district"] == d].sort_values("year_month")
        dt = pd.to_datetime(sub["year_month"].astype(str), format="%Y%m")
        fig.add_trace(go.Scatter(x=dt, y=sub["conversion_rate"], name=d, mode="lines+markers"))

    fig.add_hline(y=LEGAL_RATE, line_dash="dash", line_color="red",
                  annotation_text=f"법정 기준({LEGAL_RATE}%)")
    fig.update_layout(
        title="지역별 실질 전월세 전환율 vs 법정 기준",
        yaxis_title="전환율 (%)", height=450, hovermode="x unified"
    )
    return fig
```

---

## 🟠 Level 3 — 거시지표 연계 분석

---

### Analysis 11. 기준금리 vs 매매가 시차 상관분석

#### 목표
금리 인상이 실거래가에 반영되기까지의 평균 시차(lag)를 pearson 상관계수로 정량화한다.

#### 추가 패키지
```bash
uv pip install scipy statsmodels
```

#### 파일 위치
```
analysis/macro/11_rate_lag_corr.py
```

#### 구현 코드 명세

```python
# analysis/macro/11_rate_lag_corr.py

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def load_rate_vs_price() -> pd.DataFrame:
    """
    기준금리(한국/미국)와 서울 매매가 지수를 월별로 병합.
    반환 컬럼: ['year_month', 'bok_rate', 'fed_rate', 'avg_price', 'price_index']
    """
    macro = pd.read_parquet(Path("data/processed/macro_monthly.parquet"))
    trade = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))

    seoul = trade[trade["region"] == "서울전체"].groupby("year_month")["avg_price"].mean().reset_index()
    base  = seoul[seoul["year_month"] == 202001]["avg_price"].values[0]
    seoul["price_index"] = seoul["avg_price"] / base * 100

    merged = seoul.merge(
        macro[["year_month", "bok_rate", "fed_rate"]],
        on="year_month", how="left"
    )
    return merged


def compute_rate_lag_corr(df: pd.DataFrame, max_lag: int = 18) -> pd.DataFrame:
    results = []
    for rate_col in ["bok_rate", "fed_rate"]:
        for lag in range(0, max_lag + 1):
            shifted_rate = df[rate_col].shift(lag)
            price_chg    = df["price_index"].pct_change(3)  # 3개월 변화율
            mask = ~(shifted_rate.isna() | price_chg.isna())
            if mask.sum() < 12:
                continue
            corr, pval = pearsonr(shifted_rate[mask], price_chg[mask])
            results.append({
                "rate_type": "한국 기준금리" if rate_col == "bok_rate" else "미국 기준금리",
                "lag": lag, "corr": corr, "pvalue": pval,
                "significant": pval < 0.05
            })
    return pd.DataFrame(results)


def build_lag_corr_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("한국 기준금리 선행 효과", "미국 기준금리 선행 효과"))

    for col_idx, rate_type in enumerate(["한국 기준금리", "미국 기준금리"], start=1):
        sub = df[df["rate_type"] == rate_type]
        colors = ["#E63946" if s else "#ADB5BD" for s in sub["significant"]]
        fig.add_trace(go.Bar(
            x=sub["lag"], y=sub["corr"],
            marker_color=colors,
            name=rate_type,
            text=[f"{c:.2f}" for c in sub["corr"]],
            textposition="outside"
        ), row=1, col=col_idx)

    fig.update_xaxes(title_text="Lag (개월)")
    fig.update_yaxes(title_text="Pearson r", row=1, col=1)
    fig.update_layout(
        height=450,
        title="기준금리 → 아파트 매매가 시차 상관분석 (빨간색: p<0.05 유의)",
        showlegend=False
    )
    return fig
```

---

### Analysis 12. M2 통화량 vs 아파트 가격 분석

#### 목표
M2 YoY 증가율과 서울 매매가 YoY 상승률을 오버레이로 비교하고 유동성-가격 상관관계를 분석한다.

#### 파일 위치
```
analysis/macro/12_m2_price.py
```

#### 구현 코드 명세

```python
# analysis/macro/12_m2_price.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def load_m2_price_data() -> pd.DataFrame:
    macro = pd.read_parquet(Path("data/processed/macro_monthly.parquet"))
    trade = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))

    seoul = trade[trade["region"] == "서울전체"].groupby("year_month")["avg_price"].mean().reset_index()
    merged = seoul.merge(macro[["year_month", "m2"]], on="year_month", how="left")

    merged = merged.sort_values("year_month")
    merged["price_yoy"] = merged["avg_price"].pct_change(12) * 100
    merged["m2_yoy"]    = merged["m2"].pct_change(12) * 100
    merged["date"]      = pd.to_datetime(merged["year_month"].astype(str), format="%Y%m")

    # 국면 구분 컬럼 (M2 확장 + 가격 상승 = 과열, 등)
    def classify_phase(row):
        if pd.isna(row["price_yoy"]) or pd.isna(row["m2_yoy"]):
            return "미분류"
        if row["m2_yoy"] > 5 and row["price_yoy"] > 5:  return "과열(유동성↑·가격↑)"
        if row["m2_yoy"] < 3 and row["price_yoy"] < 0:  return "침체(유동성↓·가격↓)"
        if row["m2_yoy"] > 5 and row["price_yoy"] < 0:  return "디커플링(유동성↑·가격↓)"
        return "안정"
    merged["phase"] = merged.apply(classify_phase, axis=1)
    return merged


def build_m2_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["m2_yoy"],
        name="M2 YoY 증가율(%)", mode="lines",
        line=dict(color="#FF9800", width=2), yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price_yoy"],
        name="서울 매매가 YoY(%)", mode="lines",
        line=dict(color="#E63946", width=2.5), yaxis="y2"
    ))
    # 디커플링 구간 음영
    decouple = df[df["phase"] == "디커플링(유동성↑·가격↓)"]
    if not decouple.empty:
        fig.add_vrect(
            x0=decouple["date"].min(), x1=decouple["date"].max(),
            fillcolor="blue", opacity=0.08, layer="below",
            annotation_text="디커플링 구간"
        )
    fig.update_layout(
        title="M2 통화량 증가율 vs 서울 아파트 매매가 상승률",
        yaxis=dict(title="M2 YoY(%)", side="left"),
        yaxis2=dict(title="매매가 YoY(%)", side="right", overlaying="y"),
        height=480, hovermode="x unified",
        legend=dict(x=0.01, y=0.99)
    )
    return fig
```

---

### Analysis 13. 환율 급등 구간 Event Study

#### 목표
원달러 환율 급등 이벤트 전후 부동산 시장 반응을 Event Study 방법론으로 분석한다.

#### 파일 위치
```
analysis/macro/13_fx_event_study.py
```

#### 구현 코드 명세

```python
# analysis/macro/13_fx_event_study.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# 이벤트 윈도우 정의: 환율 급등 기준 초과 시점
FX_SHOCK_THRESHOLD = 50  # 원 기준, 전월 대비 50원 이상 급등
WINDOW_BEFORE = 3
WINDOW_AFTER  = 6

def detect_fx_shock_events(macro: pd.DataFrame) -> list:
    """월별 환율 변화에서 급등 이벤트 시점 추출."""
    macro = macro.sort_values("year_month").copy()
    macro["fx_chg"] = macro["usdkrw"].diff()
    shocks = macro[macro["fx_chg"] >= FX_SHOCK_THRESHOLD]["year_month"].tolist()
    return shocks


def compute_event_window(
    df_price: pd.DataFrame,
    df_macro: pd.DataFrame,
    event_ym: int
) -> pd.DataFrame:
    """
    이벤트 시점 기준 전후 window 내 가격/거래량 변화율 계산.
    반환: lag(-3~+6), avg_price_chg, trade_count_chg
    """
    ym_series = sorted(df_price["year_month"].unique())
    event_idx = ym_series.index(event_ym) if event_ym in ym_series else None
    if event_idx is None:
        return pd.DataFrame()

    rows = []
    for lag in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
        idx = event_idx + lag
        if 0 <= idx < len(ym_series):
            ym = ym_series[idx]
            sub = df_price[df_price["year_month"] == ym]
            rows.append({
                "lag": lag,
                "year_month": ym,
                "avg_price": sub["avg_price"].mean(),
                "trade_count": sub["trade_count"].sum()
            })
    result = pd.DataFrame(rows)
    base_price  = result[result["lag"] == 0]["avg_price"].values[0]
    base_volume = result[result["lag"] == 0]["trade_count"].values[0]
    result["price_chg_pct"]  = (result["avg_price"]   / base_price  - 1) * 100
    result["volume_chg_pct"] = (result["trade_count"] / base_volume - 1) * 100
    return result


def build_event_study_chart(all_events: list, df_price, df_macro) -> go.Figure:
    """여러 이벤트 평균 누적 반응 표시."""
    all_windows = []
    for ev in all_events:
        w = compute_event_window(df_price, df_macro, ev)
        if not w.empty:
            all_windows.append(w)

    if not all_windows:
        return go.Figure()

    combined = pd.concat(all_windows)
    avg = combined.groupby("lag")[["price_chg_pct", "volume_chg_pct"]].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=avg["lag"], y=avg["price_chg_pct"],
                             name="평균 가격 변화율(%)", mode="lines+markers",
                             line=dict(color="#E63946")))
    fig.add_trace(go.Scatter(x=avg["lag"], y=avg["volume_chg_pct"],
                             name="평균 거래량 변화율(%)", mode="lines+markers",
                             line=dict(color="#2196F3"), yaxis="y2"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="이벤트 시점")
    fig.update_layout(
        title=f"원달러 환율 급등 이벤트 전후 부동산 시장 반응 (n={len(all_events)})",
        xaxis_title="이벤트 대비 경과 월(lag)",
        yaxis=dict(title="가격 변화율(%)"),
        yaxis2=dict(title="거래량 변화율(%)", overlaying="y", side="right"),
        height=480, hovermode="x unified"
    )
    return fig
```

---

### Analysis 14. 실질 아파트 가격 지수 (CPI 디플레이트)

#### 목표
명목 매매가를 CPI로 디플레이트하여 인플레이션 효과를 제거한 실질 가격 변동을 분석한다.

#### 파일 위치
```
analysis/macro/14_real_price_index.py
```

#### 구현 코드 명세

```python
# analysis/macro/14_real_price_index.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

BASE_YM = 202001  # 2020년 1월 = 기준 100

def build_real_price_index() -> pd.DataFrame:
    """
    명목 가격 지수 = avg_price / base_price * 100
    실질 가격 지수 = 명목 지수 / (CPI / CPI_base) * 100
    반환 컬럼: ['date', 'nominal_index', 'real_index', 'cpi_index']
    """
    macro  = pd.read_parquet(Path("data/processed/macro_monthly.parquet"))
    trade  = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    seoul  = trade[trade["region"] == "서울전체"].groupby("year_month")["avg_price"].mean().reset_index()

    merged = seoul.merge(macro[["year_month", "cpi_kr"]], on="year_month", how="left").sort_values("year_month")

    base_price = merged[merged["year_month"] == BASE_YM]["avg_price"].values[0]
    base_cpi   = merged[merged["year_month"] == BASE_YM]["cpi_kr"].values[0]

    merged["nominal_index"] = merged["avg_price"] / base_price * 100
    merged["cpi_index"]     = merged["cpi_kr"]    / base_cpi   * 100
    merged["real_index"]    = merged["nominal_index"] / merged["cpi_index"] * 100
    merged["date"]          = pd.to_datetime(merged["year_month"].astype(str), format="%Y%m")
    return merged


def build_real_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["nominal_index"],
                             name="명목 가격 지수", line=dict(color="#E63946", width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["real_index"],
                             name="실질 가격 지수(CPI 조정)", line=dict(color="#2196F3", width=2.5,
                             dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["cpi_index"],
                             name="CPI 지수", line=dict(color="#FF9800", width=1.5)))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="기준(2020.01=100)")
    fig.update_layout(
        title="서울 아파트 명목 vs 실질 가격 지수 (2020.01=100)",
        yaxis_title="지수", height=480, hovermode="x unified"
    )
    return fig
```

---

### Analysis 15. 복합 상관계수 히트맵

#### 목표
매매가, 전세가, 금리, CPI, M2, 금, 유가, 환율 9개 변수 간 상관계수를 히트맵으로 표현한다.

#### 파일 위치
```
analysis/macro/15_correlation_heatmap.py
```

#### 구현 코드 명세

```python
# analysis/macro/15_correlation_heatmap.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

VARIABLES = {
    "avg_trade_price" : "매매가",
    "avg_jeonse"      : "전세가",
    "bok_rate"        : "한국금리",
    "fed_rate"        : "미국금리",
    "cpi_kr"          : "한국CPI",
    "cpi_us"          : "미국CPI",
    "m2"              : "M2",
    "gold"            : "금가격",
    "oil"             : "유가(WTI)",
    "usdkrw"          : "원달러환율",
}

def load_combined_macro(district: str = "서울전체") -> pd.DataFrame:
    macro  = pd.read_parquet(Path("data/processed/macro_monthly.parquet"))
    trade  = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    rent   = pd.read_parquet(Path("data/processed/monthly_rent_summary.parquet"))

    if district == "서울전체":
        tp = trade[trade["region"] == "서울전체"].groupby("year_month")["avg_price"].mean().reset_index()
        tp = tp.rename(columns={"avg_price": "avg_trade_price"})
        rp = (rent[(rent["region"] == "서울전체") & (rent["rent_type"] == "전세")]
              .groupby("year_month")["avg_deposit"].mean().reset_index()
              .rename(columns={"avg_deposit": "avg_jeonse"}))
    else:
        tp = trade[trade["district"] == district].groupby("year_month")["avg_price"].mean().reset_index()
        tp = tp.rename(columns={"avg_price": "avg_trade_price"})
        rp = (rent[(rent["district"] == district) & (rent["rent_type"] == "전세")]
              .groupby("year_month")["avg_deposit"].mean().reset_index()
              .rename(columns={"avg_deposit": "avg_jeonse"}))

    merged = macro.merge(tp, on="year_month", how="left").merge(rp, on="year_month", how="left")
    return merged[[c for c in VARIABLES.keys() if c in merged.columns]].dropna()


def build_corr_heatmap(df: pd.DataFrame, title_suffix: str = "서울 전체") -> go.Figure:
    corr = df.corr()
    labels = [VARIABLES[c] for c in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Pearson r")
    ))
    fig.update_layout(
        title=f"거시지표-부동산 가격 상관계수 히트맵 ({title_suffix})",
        height=550, width=700
    )
    return fig


def build_dual_heatmap(district_a: str, district_b: str) -> go.Figure:
    """두 지역 히트맵을 나란히 표시하여 민감도 차이 비교."""
    df_a = load_combined_macro(district_a)
    df_b = load_combined_macro(district_b)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=(district_a, district_b))

    for col_idx, (df, label) in enumerate([(df_a, district_a), (df_b, district_b)], start=1):
        corr = df.corr()
        xlabels = [VARIABLES.get(c, c) for c in corr.columns]
        fig.add_trace(go.Heatmap(
            z=corr.values, x=xlabels, y=xlabels,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont={"size": 9},
            showscale=(col_idx == 2)
        ), row=1, col=col_idx)

    fig.update_layout(height=520, title="지역별 거시지표 민감도 비교")
    return fig
```

---

## 🔴 Level 4 — 고급 분석 (통계 모델링 · ML)

---

### Analysis 16. 아파트 가격 예측 모델 (시계열 회귀)

#### 목표
lag된 금리·M2·거래량·전월 가격을 피처로 선형회귀 모델을 구축하고 시나리오 시뮬레이션을 지원한다.

#### 추가 패키지
```bash
uv pip install scikit-learn
```

#### 파일 위치
```
analysis/advanced/16_price_prediction.py
```

#### 구현 코드 명세

```python
# analysis/advanced/16_price_prediction.py

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
from pathlib import Path

FEATURES = {
    "bok_rate_lag1"      : "한국금리(1개월전)",
    "bok_rate_lag3"      : "한국금리(3개월전)",
    "m2_yoy_lag2"        : "M2증가율(2개월전)",
    "trade_count_lag1"   : "거래량(1개월전)",
    "price_lag1"         : "전월매매가",
    "price_lag12"        : "전년동월매매가",
    "cpi_kr_lag1"        : "한국CPI(1개월전)",
}

def build_feature_matrix() -> pd.DataFrame:
    macro = pd.read_parquet(Path("data/processed/macro_monthly.parquet"))
    trade = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))

    seoul = (trade[trade["region"] == "서울전체"]
             .groupby("year_month")
             .agg(avg_price=("avg_price","mean"), trade_count=("trade_count","sum"))
             .reset_index().sort_values("year_month"))

    df = seoul.merge(macro[["year_month","bok_rate","m2","cpi_kr"]], on="year_month", how="left")
    df["m2_yoy"] = df["m2"].pct_change(12) * 100

    # Lag 피처 생성
    df["bok_rate_lag1"]    = df["bok_rate"].shift(1)
    df["bok_rate_lag3"]    = df["bok_rate"].shift(3)
    df["m2_yoy_lag2"]      = df["m2_yoy"].shift(2)
    df["trade_count_lag1"] = df["trade_count"].shift(1)
    df["price_lag1"]       = df["avg_price"].shift(1)
    df["price_lag12"]      = df["avg_price"].shift(12)
    df["cpi_kr_lag1"]      = df["cpi_kr"].shift(1)
    df["target"]           = df["avg_price"]

    return df.dropna()


def train_model(df: pd.DataFrame):
    X = df[list(FEATURES.keys())].values
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    model = Ridge(alpha=1.0)
    mape_scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        pred = model.predict(X_scaled[val_idx])
        mape_scores.append(mean_absolute_percentage_error(y[val_idx], pred))

    model.fit(X_scaled, y)
    coef_df = pd.DataFrame({
        "feature": list(FEATURES.values()),
        "coef": model.coef_
    }).sort_values("coef", key=abs, ascending=False)

    return model, scaler, coef_df, np.mean(mape_scores)


def simulate_scenario(
    model, scaler,
    bok_rate: float,
    m2_yoy: float,
    trade_count: int,
    price_lag1: float,
    price_lag12: float,
    cpi_kr: float
) -> float:
    """슬라이더로 입력받은 시나리오 값으로 다음 달 예상가 반환."""
    X = np.array([[bok_rate, bok_rate, m2_yoy, trade_count,
                   price_lag1, price_lag12, cpi_kr]])
    return model.predict(scaler.transform(X))[0]


def build_prediction_chart(df: pd.DataFrame, model, scaler) -> go.Figure:
    X_scaled = scaler.transform(df[list(FEATURES.keys())].values)
    y_pred   = model.predict(X_scaled)
    dates    = pd.to_datetime(df["year_month"].astype(str), format="%Y%m")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=df["target"], name="실제 매매가", line=dict(color="#E63946")))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, name="모델 예측가", line=dict(color="#2196F3", dash="dot")))
    fig.update_layout(title="Ridge 회귀 모델 — 서울 아파트 매매가 예측 vs 실제",
                      yaxis_title="평균 매매가(만원)", height=480, hovermode="x unified")
    return fig
```

---

### Analysis 17. 지역 군집 분석 (DTW Clustering)

#### 목표
자치구별 가격 시계열 패턴을 DTW 거리로 군집화하여 가격 흐름이 유사한 지역 그룹을 식별한다.

#### 추가 패키지
```bash
uv pip install dtaidistance scikit-learn
```

#### 파일 위치
```
analysis/advanced/17_dtw_clustering.py
```

#### 구현 코드 명세

```python
# analysis/advanced/17_dtw_clustering.py

import pandas as pd
import numpy as np
from dtaidistance import dtw_ndim, clustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

N_CLUSTERS = 4
CLUSTER_NAMES = {
    0: "강남 동조화 그룹",
    1: "중산권 독립 그룹",
    2: "외곽 후행 그룹",
    3: "경기 신도시 그룹"
}

def load_district_price_matrix() -> tuple[pd.DataFrame, list]:
    """
    자치구별 월별 평균 가격을 정규화한 시계열 행렬 반환.
    반환: (matrix: shape [n_districts, n_months], districts: list)
    """
    df = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    pivot = df.pivot_table(index="year_month", columns="district", values="avg_price")
    pivot = pivot.sort_index().fillna(method="ffill").fillna(method="bfill")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pivot)  # shape: [n_months, n_districts]
    return scaled.T, list(pivot.columns)   # transpose: [n_districts, n_months]


def compute_dtw_clustering(matrix: np.ndarray) -> np.ndarray:
    """DTW 거리 행렬 계산 후 계층적 군집화."""
    dist_matrix = dtw_ndim.distance_matrix_fast(matrix)
    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="precomputed",
        linkage="average"
    )
    labels = model.fit_predict(dist_matrix)
    return labels


def build_cluster_map(districts: list, labels: np.ndarray) -> go.Figure:
    """지도 위에 군집 색상으로 자치구 표시 (folium 대신 plotly scattermapbox)."""
    # 자치구 대표 좌표 (config/settings.py 에서 관리 권장)
    COORDS = {
        "강남구": (37.5172, 127.0473), "서초구": (37.4837, 127.0324),
        "용산구": (37.5311, 126.9810), "마포구": (37.5663, 126.9017),
        "노원구": (37.6542, 127.0568), "도봉구": (37.6688, 127.0471),
        # ... 전체 자치구 좌표는 config/settings.py에 정의
    }
    cluster_df = pd.DataFrame({"district": districts, "cluster": labels})
    cluster_df["cluster_name"] = cluster_df["cluster"].map(CLUSTER_NAMES)
    cluster_df["lat"] = cluster_df["district"].map(lambda d: COORDS.get(d, (37.5, 127.0))[0])
    cluster_df["lon"] = cluster_df["district"].map(lambda d: COORDS.get(d, (37.5, 127.0))[1])

    fig = px.scatter_mapbox(
        cluster_df, lat="lat", lon="lon",
        color="cluster_name", hover_name="district",
        zoom=10, mapbox_style="carto-positron",
        title="아파트 가격 시계열 DTW 군집 분류",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(marker=dict(size=18))
    fig.update_layout(height=550)
    return fig


def build_cluster_timeseries(
    matrix: np.ndarray,
    districts: list,
    labels: np.ndarray
) -> go.Figure:
    """군집별 평균 시계열 라인 차트."""
    n_months = matrix.shape[1]
    months = list(range(n_months))

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for c in range(N_CLUSTERS):
        idx = np.where(labels == c)[0]
        avg = matrix[idx].mean(axis=0)
        fig.add_trace(go.Scatter(
            x=months, y=avg,
            name=CLUSTER_NAMES.get(c, f"군집{c}"),
            line=dict(color=colors[c], width=2.5)
        ))
    fig.update_layout(
        title="군집별 평균 가격 시계열 (정규화)",
        xaxis_title="월 인덱스(2020.01~)", yaxis_title="정규화 가격",
        height=420, hovermode="x unified"
    )
    return fig
```

---

### Analysis 18. 이상 거래 감지 (Anomaly Detection)

#### 목표
Isolation Forest로 비정상적 고가/저가 거래를 자동 탐지하고 산점도로 시각화한다.

#### 추가 패키지
```bash
uv pip install scikit-learn
```

#### 파일 위치
```
analysis/advanced/18_anomaly_detection.py
```

#### 구현 코드 명세

```python
# analysis/advanced/18_anomaly_detection.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import glob
from pathlib import Path

CONTAMINATION = 0.03  # 이상 거래 비율 가정 (3%)

def load_raw_transactions(district_code: str) -> pd.DataFrame:
    """
    특정 자치구의 전체 원시 거래 데이터 로드.
    피처: price_per_sqm, area, floor, build_age
    """
    files = glob.glob(f"data/raw/molit/apt_trade/{district_code}*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df["price"]         = df["거래금액"].str.replace(",","").astype(float)
    df["area"]          = df["전용면적"].astype(float)
    df["floor"]         = df["층"].astype(float)
    df["build_age"]     = df["년"].astype(int) - df["건축년도"].astype(int)
    df["price_per_sqm"] = df["price"] / df["area"]
    df["date"]          = pd.to_datetime(
        df["년"].astype(str) + df["월"].astype(str).str.zfill(2) + "01"
    )
    return df.dropna(subset=["price_per_sqm", "area", "floor", "build_age"])


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    피처 기반 Isolation Forest 이상 탐지.
    반환: df에 'is_anomaly', 'anomaly_score' 컬럼 추가
    """
    features = df[["price_per_sqm", "area", "floor", "build_age"]].copy()
    # 로그 변환으로 스케일 정규화
    features["log_price"] = np.log1p(features["price_per_sqm"])
    features["log_area"]  = np.log1p(features["area"])

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    model.fit(features[["log_price", "log_area", "floor", "build_age"]])
    df = df.copy()
    df["anomaly_score"] = model.score_samples(
        features[["log_price", "log_area", "floor", "build_age"]]
    )
    df["is_anomaly"] = model.predict(
        features[["log_price", "log_area", "floor", "build_age"]]
    ) == -1
    return df


def build_anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """
    x: 거래일자 / y: ㎡당 가격 / 색: 이상여부
    이상 거래에 빨간 마커 표시
    """
    normal  = df[~df["is_anomaly"]]
    anomaly = df[df["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["price_per_sqm"],
        mode="markers", name="정상 거래",
        marker=dict(color="lightblue", size=4, opacity=0.5)
    ))
    fig.add_trace(go.Scatter(
        x=anomaly["date"], y=anomaly["price_per_sqm"],
        mode="markers", name="이상 거래",
        marker=dict(color="red", size=8, symbol="x"),
        text=anomaly["아파트"] + " " + anomaly["area"].astype(str) + "㎡",
        hovertemplate="%{text}<br>㎡당: %{y:,.0f}만원<extra></extra>"
    ))
    fig.update_layout(
        title="Isolation Forest 이상 거래 탐지 (빨간 X = 이상 거래)",
        xaxis_title="거래일자", yaxis_title="㎡당 가격(만원)",
        height=500
    )
    return fig
```

---

### Analysis 19. 학군·GTX 인프라 효과 분석 (이중차분법)

#### 목표
지하철 신설 노선(GTX-A 등) 발표 전후 인근 지역 가격 변화를 Difference-in-Differences로 계량화한다.

#### 추가 패키지
```bash
uv pip install statsmodels linearmodels
```

#### 파일 위치
```
analysis/advanced/19_did_analysis.py
```

#### 구현 코드 명세

```python
# analysis/advanced/19_did_analysis.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from pathlib import Path

# 이벤트 설정 예시 (config/settings.py에서 관리 권장)
DID_EVENTS = {
    "GTX-A_착공발표": {
        "event_ym": 202012,
        "treatment_districts": ["고양시 덕양구", "파주시"],
        "control_districts":   ["의정부시", "구리시"],
        "description": "GTX-A 노선 착공 발표 (2020.12)"
    },
    "GTX-B_예타통과": {
        "event_ym": 202106,
        "treatment_districts": ["인천시 부평구", "남양주시"],
        "control_districts":   ["부천시", "하남시"],
        "description": "GTX-B 예비타당성 통과 (2021.06)"
    }
}

def build_did_dataset(event_key: str) -> pd.DataFrame:
    """
    처리군(treatment)/통제군(control) + 이벤트 전후(post) 패널 구성.
    DiD 식: price ~ treatment + post + treatment:post + controls
    """
    event = DID_EVENTS[event_key]
    event_ym = event["event_ym"]

    df = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))
    macro = pd.read_parquet(Path("data/processed/macro_monthly.parquet"))

    all_districts = event["treatment_districts"] + event["control_districts"]
    sub = df[df["district"].isin(all_districts)].copy()
    sub = sub.merge(macro[["year_month", "bok_rate"]], on="year_month", how="left")

    sub["treatment"] = sub["district"].isin(event["treatment_districts"]).astype(int)
    sub["post"]      = (sub["year_month"] >= event_ym).astype(int)
    sub["log_price"] = np.log(sub["avg_price"])

    # 이벤트 ±12개월 윈도우만 사용
    ym_arr = sorted(sub["year_month"].unique())
    ev_idx = ym_arr.index(event_ym) if event_ym in ym_arr else len(ym_arr)//2
    window_ym = ym_arr[max(0, ev_idx-12) : min(len(ym_arr), ev_idx+13)]
    return sub[sub["year_month"].isin(window_ym)]


def run_did_regression(df: pd.DataFrame) -> dict:
    """
    OLS DiD 회귀 실행.
    반환: {summary_text, did_coef, did_pvalue, did_ci}
    """
    result = smf.ols(
        "log_price ~ treatment + post + treatment:post + bok_rate",
        data=df
    ).fit(cov_type="HC3")

    did_coef  = result.params.get("treatment:post", np.nan)
    did_pval  = result.pvalues.get("treatment:post", np.nan)
    did_ci    = result.conf_int().loc["treatment:post"].tolist() if "treatment:post" in result.params else [np.nan, np.nan]
    pct_effect = (np.exp(did_coef) - 1) * 100 if not np.isnan(did_coef) else np.nan

    return {
        "summary": result.summary().as_text(),
        "did_coef": did_coef,
        "did_pvalue": did_pval,
        "did_ci": did_ci,
        "pct_effect": pct_effect,
        "n_obs": result.nobs
    }


def build_parallel_trend_chart(df: pd.DataFrame, event_ym: int) -> go.Figure:
    """이중차분법 평행추세 가정 확인 차트."""
    trend = df.groupby(["year_month", "treatment"])["avg_price"].mean().reset_index()
    trend["date"] = pd.to_datetime(trend["year_month"].astype(str), format="%Y%m")

    fig = go.Figure()
    for grp, label, color in [(1,"처리군(인근지역)","#E63946"),(0,"통제군(비교지역)","#457B9D")]:
        sub = trend[trend["treatment"] == grp]
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["avg_price"],
                                 name=label, mode="lines+markers",
                                 line=dict(color=color, width=2.5)))
    fig.add_vline(
        x=pd.Timestamp(str(event_ym)[:4] + "-" + str(event_ym)[4:] + "-01"),
        line_dash="dash", line_color="gray",
        annotation_text="이벤트 시점"
    )
    fig.update_layout(
        title="DiD 평행추세 확인 (처리군 vs 통제군)",
        yaxis_title="평균 매매가(만원)",
        height=420, hovermode="x unified"
    )
    return fig
```

---

### Analysis 20. 거시지표 기반 부동산 사이클 국면 분류

#### 목표
기준금리 방향·M2 증가율·거래량 추세 3개 신호로 시장 국면을 4단계로 분류하고 현재 국면을 실시간 표시한다.

#### 추가 패키지
```bash
uv pip install hmmlearn
```

#### 파일 위치
```
analysis/advanced/20_market_cycle.py
```

#### 구현 코드 명세

```python
# analysis/advanced/20_market_cycle.py

import pandas as pd
import numpy as np
from hmmlearn import hmm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

PHASE_COLORS = {
    "과열": "#E63946",
    "조정": "#FF9800",
    "침체": "#607D8B",
    "회복": "#4CAF50"
}

# ─── 방법 A: 규칙 기반 국면 분류 ────────────────────────────────────────────

def classify_rule_based(row: pd.Series) -> str:
    """
    규칙 기반 국면 분류.
    rate_dir: 1(인상) / 0(동결) / -1(인하)
    m2_yoy: M2 전년비 증가율
    vol_mom: 거래량 전월비
    """
    rate  = row.get("rate_direction", 0)
    m2    = row.get("m2_yoy", 0)
    vol   = row.get("vol_mom_3ma", 0)   # 3개월 이동평균

    if   rate <= 0 and m2 >= 5 and vol >= 0:   return "과열"
    elif rate >= 1 and m2 <= 3 and vol <= -5:  return "침체"
    elif rate >= 1 and vol <= 0:               return "조정"
    else:                                       return "회복"


# ─── 방법 B: HMM 기반 국면 분류 ─────────────────────────────────────────────

def train_hmm_cycle(df: pd.DataFrame) -> tuple:
    """
    3개 시그널로 4-state Gaussian HMM 학습.
    시그널: [금리변화방향, M2_YoY, 거래량_MoM]
    반환: (model, state_sequence, state_label_map)
    """
    features = df[["rate_direction", "m2_yoy", "vol_mom_3ma"]].dropna().values

    model = hmm.GaussianHMM(
        n_components=4,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )
    model.fit(features)
    states = model.predict(features)

    # 상태별 평균 가격 변화율로 레이블 자동 할당
    state_price_chg = {}
    for s in range(4):
        idx = np.where(states == s)[0]
        state_price_chg[s] = df["price_yoy"].iloc[idx].mean() if len(idx) > 0 else 0

    sorted_states = sorted(state_price_chg, key=state_price_chg.get, reverse=True)
    label_map = {
        sorted_states[0]: "과열",
        sorted_states[1]: "회복",
        sorted_states[2]: "조정",
        sorted_states[3]: "침체",
    }
    return model, states, label_map


# ─── 피처 데이터 준비 ─────────────────────────────────────────────────────────

def load_cycle_features() -> pd.DataFrame:
    macro  = pd.read_parquet(Path("data/processed/macro_monthly.parquet"))
    trade  = pd.read_parquet(Path("data/processed/monthly_trade_summary.parquet"))

    seoul = (trade[trade["region"] == "서울전체"]
             .groupby("year_month")
             .agg(avg_price=("avg_price","mean"), trade_count=("trade_count","sum"))
             .reset_index().sort_values("year_month"))

    df = seoul.merge(macro[["year_month","bok_rate","m2"]], on="year_month", how="left")
    df["rate_direction"]  = np.sign(df["bok_rate"].diff())  # 1/0/-1
    df["m2_yoy"]          = df["m2"].pct_change(12) * 100
    df["vol_mom"]         = df["trade_count"].pct_change() * 100
    df["vol_mom_3ma"]     = df["vol_mom"].rolling(3).mean()
    df["price_yoy"]       = df["avg_price"].pct_change(12) * 100

    # 규칙 기반 분류
    df["phase_rule"] = df.apply(classify_rule_based, axis=1)
    df["date"]       = pd.to_datetime(df["year_month"].astype(str), format="%Y%m")
    return df.dropna()


# ─── 시각화 ──────────────────────────────────────────────────────────────────

def build_cycle_dashboard(df: pd.DataFrame, use_hmm: bool = False) -> go.Figure:
    """
    상단: 서울 매매가 + 국면 배경 음영
    하단: 3개 시그널 서브플롯 (금리방향, M2, 거래량)
    """
    if use_hmm:
        _, states, label_map = train_hmm_cycle(df)
        df = df.copy()
        df["phase"] = [label_map.get(s, "미분류") for s in states[:len(df)]]
    else:
        df["phase"] = df["phase_rule"]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("서울 매매가 + 국면", "기준금리 방향", "M2 YoY(%)", "거래량 전월비(%)"),
        row_heights=[0.4, 0.2, 0.2, 0.2], vertical_spacing=0.04
    )

    # 매매가 라인
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_price"],
        name="서울 평균 매매가", line=dict(color="#212121", width=2)
    ), row=1, col=1)

    # 국면 배경 음영
    phase_changes = df[df["phase"] != df["phase"].shift()].index.tolist() + [df.index[-1]]
    for i in range(len(phase_changes) - 1):
        start_idx = phase_changes[i]
        end_idx   = phase_changes[i + 1]
        phase     = df.loc[start_idx, "phase"]
        fig.add_vrect(
            x0=df.loc[start_idx, "date"],
            x1=df.loc[end_idx if end_idx < len(df) else end_idx-1, "date"],
            fillcolor=PHASE_COLORS.get(phase, "lightgray"),
            opacity=0.15, layer="below",
            annotation_text=phase if i % 3 == 0 else "",
            annotation_position="top left",
            row=1, col=1
        )

    # 서브 시그널
    fig.add_trace(go.Bar(x=df["date"], y=df["rate_direction"],
                         marker_color=["#E63946" if v > 0 else "#4CAF50" if v < 0 else "#9E9E9E"
                                       for v in df["rate_direction"]],
                         name="금리방향"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["m2_yoy"],
                             line=dict(color="#FF9800"), name="M2 YoY"), row=3, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["vol_mom_3ma"],
                         marker_color=["#457B9D" if v > 0 else "#E63946"
                                       for v in df["vol_mom_3ma"].fillna(0)],
                         name="거래량3MA"), row=4, col=1)

    fig.update_layout(
        height=800,
        title=f"부동산 시장 국면 분류 ({'HMM' if use_hmm else '규칙 기반'})",
        showlegend=False
    )
    return fig


def get_current_phase(df: pd.DataFrame) -> dict:
    """대시보드 상단 배너용 현재 국면 정보 반환."""
    latest = df.sort_values("year_month").iloc[-1]
    return {
        "phase": latest["phase_rule"],
        "color": PHASE_COLORS.get(latest["phase_rule"], "gray"),
        "bok_rate": latest["bok_rate"],
        "m2_yoy": round(latest["m2_yoy"], 2),
        "vol_mom": round(latest["vol_mom"], 2),
        "year_month": latest["year_month"]
    }
```

#### Streamlit 배너 컴포넌트
```python
# dashboard/components/cycle_banner.py
import streamlit as st
from analysis.advanced.market_cycle import load_cycle_features, get_current_phase

def render_cycle_banner():
    df     = load_cycle_features()
    info   = get_current_phase(df)
    ym_str = str(info["year_month"])
    label  = f"{ym_str[:4]}년 {ym_str[4:]}월"

    st.markdown(f"""
    <div style="background:{info['color']}22; border-left:5px solid {info['color']};
                padding:12px 20px; border-radius:6px; margin-bottom:20px;">
        <span style="font-size:22px; font-weight:700; color:{info['color']}">
            📊 현재 시장 국면: {info['phase']}
        </span>
        &nbsp;&nbsp;
        <span style="color:#555; font-size:14px">
            ({label} 기준 | 금리 {info['bok_rate']}% | M2 {info['m2_yoy']:+.1f}% | 거래량 {info['vol_mom']:+.1f}%)
        </span>
    </div>
    """, unsafe_allow_html=True)
```

---

## 부록 A. 분석 모듈 디렉토리 구조 (최종)

```
analysis/
├── __init__.py
├── basic/
│   ├── __init__.py
│   ├── 01_monthly_volume.py
│   ├── 02_district_ranking.py
│   ├── 03_area_price_dist.py
│   ├── 04_age_premium.py
│   └── 05_jeonse_ratio.py
├── intermediate/
│   ├── __init__.py
│   ├── 06_heatmap.py
│   ├── 07_floor_premium.py
│   ├── 08_yoy_map.py
│   ├── 09_volume_price_lag.py
│   └── 10_conversion_rate.py
├── macro/
│   ├── __init__.py
│   ├── 11_rate_lag_corr.py
│   ├── 12_m2_price.py
│   ├── 13_fx_event_study.py
│   ├── 14_real_price_index.py
│   └── 15_correlation_heatmap.py
└── advanced/
    ├── __init__.py
    ├── 16_price_prediction.py
    ├── 17_dtw_clustering.py
    ├── 18_anomaly_detection.py
    ├── 19_did_analysis.py
    └── 20_market_cycle.py
```

---

## 부록 B. 전체 패키지 `requirements.txt` (uv 기준)

```
# ── 데이터 수집 ──────────────────────────
requests==2.31.0
yfinance==0.2.36
python-dotenv==1.0.0

# ── 데이터 처리 ──────────────────────────
pandas==2.1.4
numpy==1.26.4
pyarrow==15.0.0

# ── 시각화 ───────────────────────────────
plotly==5.18.0
folium==0.16.0
streamlit-folium==0.18.0

# ── 지도/공간 ────────────────────────────
geopandas==0.14.3
pyproj==3.6.1

# ── 대시보드 ─────────────────────────────
streamlit==1.32.0

# ── 통계/ML (Level 3~4) ──────────────────
scipy==1.12.0
statsmodels==0.14.1
scikit-learn==1.4.1
linearmodels==5.4

# ── 고급 분석 (Level 4) ──────────────────
dtaidistance==2.3.11
hmmlearn==0.3.2

# ── 유틸리티 ─────────────────────────────
tqdm==4.66.1
loguru==0.7.2

# ── 개발/테스트 ──────────────────────────
pytest==8.0.0
black==24.3.0
ruff==0.3.4
```

### uv 일괄 설치 명령
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 부록 C. Cursor 프롬프트 활용 팁

각 분석을 Cursor에게 지시할 때 다음 템플릿을 사용하세요:

```
[분석 N 구현 요청]
`analysis/{category}/NN_*.py` 파일을 생성해줘.

참고 명세:
- 함수 목록: [위 명세의 함수명 열거]
- 입력 데이터: data/processed/ 또는 data/raw/molit/
- 반환 타입: plotly Figure 또는 pd.DataFrame
- 의존 패키지: [패키지명] (이미 .venv에 설치됨)

추가 조건:
- @st.cache_data 데코레이터 불필요 (분석 모듈은 순수 함수로만 작성)
- 모든 파일 경로는 pathlib.Path 사용
- 예외 처리: 데이터 파일 없을 시 빈 DataFrame 반환 후 로그 출력
- 타입 힌트 필수
```

---

*본 가이드는 `real_estate_project_plan.md`의 파이프라인 구조 완성을 전제로 합니다.  
Level 1 → Level 4 순서로 진행하며, 각 분석 완료 후 `dashboard/pages/`에 Streamlit 컴포넌트를 통합하세요.*
