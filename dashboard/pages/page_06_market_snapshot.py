"""Page 06 – A. 데이터 진단 & 시장 스냅샷.

A-1 월별 거래량·중위 ㎡당 가격·분산 추이
A-2 면적 믹스 변화 & 구성효과 분해
A-3 이상치·오류·비정상 거래 탐지
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.data_loader import (
    load_snapshot_monthly_trade,
    load_snapshot_monthly_rent,
    load_snapshot_area_mix,
    load_snapshot_outliers,
)
from config.settings import SEOUL_REGIONS, GYEONGGI_REGIONS


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def _region_options() -> dict[str, str]:
    """지역 선택 옵션 딕셔너리를 반환한다."""
    options = {"ALL": "전체", "SEOUL": "서울 전체", "GYEONGGI": "경기 전체"}
    options.update(SEOUL_REGIONS)
    options.update(GYEONGGI_REGIONS)
    return options


def _plotly_line(df: pd.DataFrame, x: str, y_cols: list[str], title: str,
                 y_label: str = "", colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    palette = colors or px.colors.qualitative.Plotly
    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col],
            name=col,
            mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="연월",
        yaxis_title=y_label,
        legend=dict(orientation="h", y=-0.2),
        height=380,
    )
    fig.update_xaxes(tickformat="%Y-%m", dtick="M6", tickangle=-30)
    return fig


# ---------------------------------------------------------------------------
# A-1: 월별 거래량·가격 추이
# ---------------------------------------------------------------------------

def _render_a1(trade_df: pd.DataFrame, rent_df: pd.DataFrame, selected_code: str) -> None:
    st.subheader("A-1. 월별 거래량·중위 ㎡당 가격·분산 추이")

    # 지역 필터링
    t = trade_df[trade_df["sggCd"] == selected_code].sort_values("month")
    r = rent_df[rent_df["sggCd"] == selected_code].sort_values("month") if not rent_df.empty else pd.DataFrame()

    if t.empty:
        st.warning("선택한 지역의 매매 데이터가 없습니다.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("총 매매건수", f"{t['trade_count'].sum():,}")
    latest = t.iloc[-1]
    col2.metric(
        "최근월 중위 ㎡당 가격",
        f"{latest['price_median_m2']:,.0f}만원/㎡",
    )
    if len(t) >= 13:
        yoy = latest["price_median_m2"] / t.iloc[-13]["price_median_m2"] - 1
        col3.metric("전년 동월 대비", f"{yoy:+.1%}")
    else:
        col3.metric("전년 동월 대비", "N/A")

    # 거래량 바 차트
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=t["month"], y=t["trade_count"],
        name="매매 거래량",
        marker_color="steelblue", opacity=0.75,
    ))
    if not r.empty:
        r_jeonse = r[r["rentType"] == "전세"]
        r_wolse = r[r["rentType"] == "월세"]
        if not r_jeonse.empty:
            fig_vol.add_trace(go.Bar(
                x=r_jeonse["month"], y=r_jeonse["rent_count"],
                name="전세 거래량", marker_color="royalblue", opacity=0.6,
            ))
        if not r_wolse.empty:
            fig_vol.add_trace(go.Bar(
                x=r_wolse["month"], y=r_wolse["rent_count"],
                name="월세 거래량", marker_color="darkorange", opacity=0.6,
            ))
    fig_vol.update_layout(
        title="월별 거래건수 (매매·전세·월세)",
        barmode="group",
        xaxis_tickformat="%Y-%m", xaxis_dtick="M6", xaxis_tickangle=-30,
        height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_vol, width="stretch")

    # 중위 ㎡당 가격 + 이동평균
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=t["month"], y=t["price_median_m2"],
        name="중위 ㎡당 가격", mode="lines",
        line=dict(color="crimson", width=2),
    ))
    # 신뢰구간 (IQR)
    if "price_p25_m2" in t.columns and "price_p75_m2" in t.columns:
        fig_price.add_trace(go.Scatter(
            x=pd.concat([t["month"], t["month"].iloc[::-1]]),
            y=pd.concat([t["price_p75_m2"], t["price_p25_m2"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(220,20,60,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IQR (25%~75%)", showlegend=True,
        ))
    # 이동평균
    for col, label, color in [
        ("rolling_3m_median_m2", "3개월 이동평균", "orange"),
        ("rolling_12m_median_m2", "12개월 이동평균", "navy"),
    ]:
        if col in t.columns:
            fig_price.add_trace(go.Scatter(
                x=t["month"], y=t[col],
                name=label, mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
            ))
    fig_price.update_layout(
        title="중위 ㎡당 가격 추이 (만원/㎡)",
        xaxis_tickformat="%Y-%m", xaxis_dtick="M6", xaxis_tickangle=-30,
        height=400,
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig_price, width="stretch")

    # 가격 분산 (표준편차)
    if "price_std_m2" in t.columns:
        fig_std = go.Figure()
        fig_std.add_trace(go.Scatter(
            x=t["month"], y=t["price_std_m2"],
            name="표준편차", mode="lines+markers",
            line=dict(color="purple", width=1.5),
            marker=dict(size=3),
        ))
        fig_std.update_layout(
            title="월별 ㎡당 가격 표준편차 (분산 추이)",
            xaxis_tickformat="%Y-%m", xaxis_dtick="M6", xaxis_tickangle=-30,
            height=280,
        )
        st.plotly_chart(fig_std, width="stretch")


# ---------------------------------------------------------------------------
# A-2: 면적 믹스 변화
# ---------------------------------------------------------------------------

def _render_a2(area_mix_df: pd.DataFrame, selected_code: str) -> None:
    st.subheader("A-2. 면적 믹스 변화 & 구성효과 분해")

    df = area_mix_df[area_mix_df["sggCd"] == selected_code].copy()
    if df.empty:
        st.warning("선택한 지역의 면적 믹스 데이터가 없습니다.")
        return

    bucket_order = ["~60㎡", "60~85㎡", "85~102㎡", "102㎡~"]

    # Stacked area chart (면적 비중)
    share_pivot = (
        df.pivot_table(index="month", columns="area_bucket", values="share_pct", aggfunc="sum")
        .reindex(columns=bucket_order)
        .reset_index()
    )

    fig_mix = go.Figure()
    colors_area = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, bucket in enumerate(bucket_order):
        if bucket not in share_pivot.columns:
            continue
        fig_mix.add_trace(go.Scatter(
            x=share_pivot["month"],
            y=share_pivot[bucket],
            name=bucket,
            mode="lines",
            stackgroup="one",
            fillcolor=colors_area[i],
            line=dict(color=colors_area[i], width=0),
        ))
    fig_mix.update_layout(
        title="면적 구간별 거래 비중 추이 (%)",
        yaxis=dict(title="비중 (%)", range=[0, 100]),
        xaxis_tickformat="%Y-%m", xaxis_dtick="M6", xaxis_tickangle=-30,
        height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_mix, width="stretch")

    # 면적 구간별 중위 ㎡당 가격
    fig_price_area = go.Figure()
    for i, bucket in enumerate(bucket_order):
        sub = df[df["area_bucket"] == bucket].sort_values("month")
        if sub.empty:
            continue
        fig_price_area.add_trace(go.Scatter(
            x=sub["month"], y=sub["price_median_m2"],
            name=bucket, mode="lines",
            line=dict(color=colors_area[i], width=2),
        ))
    fig_price_area.update_layout(
        title="면적 구간별 중위 ㎡당 가격 추이 (만원/㎡)",
        xaxis_tickformat="%Y-%m", xaxis_dtick="M6", xaxis_tickangle=-30,
        height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_price_area, width="stretch")

    # 구성효과 분해
    comp_cols = ["actual_mean_m2", "fixed_weight_mean_m2", "composition_effect_m2"]
    if all(c in df.columns for c in comp_cols):
        comp_monthly = (
            df.groupby("month")[comp_cols]
            .first()
            .reset_index()
            .sort_values("month")
        )

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=comp_monthly["month"], y=comp_monthly["actual_mean_m2"],
            name="실제 평균가", mode="lines", line=dict(color="crimson", width=2),
        ))
        fig_comp.add_trace(go.Scatter(
            x=comp_monthly["month"], y=comp_monthly["fixed_weight_mean_m2"],
            name="고정가중 평균 (2020 기준)", mode="lines",
            line=dict(color="navy", width=2, dash="dash"),
        ))
        fig_comp.update_layout(
            title="실제 평균 vs 고정가중 평균 (구성효과 분리)",
            xaxis_tickformat="%Y-%m", xaxis_dtick="M6", xaxis_tickangle=-30,
            height=320,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_comp, width="stretch")

        fig_effect = go.Figure()
        fig_effect.add_trace(go.Bar(
            x=comp_monthly["month"],
            y=comp_monthly["composition_effect_m2"],
            name="구성효과",
            marker_color=[
                "crimson" if v > 0 else "steelblue"
                for v in comp_monthly["composition_effect_m2"]
            ],
            opacity=0.75,
        ))
        fig_effect.add_hline(y=0, line_color="black", line_width=1)
        fig_effect.update_layout(
            title="월별 면적 구성효과 (만원/㎡) — 양수: 대형 비중 증가가 평균가격을 끌어올림",
            xaxis_tickformat="%Y-%m", xaxis_dtick="M6", xaxis_tickangle=-30,
            height=300,
        )
        st.plotly_chart(fig_effect, width="stretch")

        # 구성효과 요약
        total_effect_pct = (
            comp_monthly["composition_effect_m2"].mean()
            / comp_monthly["actual_mean_m2"].mean()
            * 100
        )
        st.info(
            f"전체 기간 평균 구성효과: **{comp_monthly['composition_effect_m2'].mean():+.1f}만원/㎡** "
            f"(전체 평균가의 {total_effect_pct:+.1f}%)\n\n"
            "→ 양수면 고가 대형 평형 거래가 늘어나 평균가가 실질보다 높게 보임을 의미합니다."
        )


# ---------------------------------------------------------------------------
# A-3: 이상치 탐지
# ---------------------------------------------------------------------------

def _render_a3(outliers_df: pd.DataFrame, selected_code: str) -> None:
    st.subheader("A-3. 이상치·오류·비정상 거래 탐지")
    st.caption(
        "탐지 기준: 같은 단지(aptSeq) × 면적유형(area_repr) 기준, "
        "직전 6개월 이내 월별 평균 시세 대비 ±25% 초과 거래 "
        "| 사전 제외: 1층 거래, 직전 유효 거래 대비 연속 등락 25% 초과"
    )

    if outliers_df.empty:
        st.info("이상치 데이터가 없습니다. 파이프라인을 먼저 실행해주세요.")
        st.code("uv run python pipelines/market_snapshot_pipeline.py", language="bash")
        return

    # 지역 필터 (dong_repr 컬럼에서 sggCd 추출)
    df = outliers_df.copy()
    if "dong_repr" in df.columns:
        df["_sggCd"] = df["dong_repr"].str.extract(r"\((\d+)\)")
    elif "aptSeq" in df.columns:
        df["_sggCd"] = df["aptSeq"].astype(str).str.split("-").str[0]
    else:
        df["_sggCd"] = "UNKNOWN"

    from config.settings import SEOUL_REGIONS as _SEOUL_REGIONS
    if selected_code not in ("ALL", "SEOUL", "GYEONGGI"):
        df = df[df["_sggCd"] == selected_code]
    elif selected_code == "SEOUL":
        df = df[df["_sggCd"].isin(_SEOUL_REGIONS.keys())]
    elif selected_code == "GYEONGGI":
        df = df[~df["_sggCd"].isin(_SEOUL_REGIONS.keys())]

    if df.empty:
        st.warning("선택 지역의 이상치가 없습니다.")
        return

    # KPI
    high_cnt = (df["outlier_direction"] == "고가이상치").sum() if "outlier_direction" in df.columns else 0
    low_cnt  = (df["outlier_direction"] == "저가이상치").sum() if "outlier_direction" in df.columns else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("총 이상치 건수", f"{len(df):,}")
    col2.metric("고가 이상치 (↑)", f"{high_cnt:,}")
    col3.metric("저가 이상치 (↓)", f"{low_cnt:,}")

    # 월별 이상치 건수 추이 (고가/저가 분리)
    if "outlier_direction" in df.columns:
        monthly_ct = (
            df.groupby(["month", "outlier_direction"])
            .size()
            .reset_index(name="count")
            .sort_values("month")
        )
        fig_monthly = px.bar(
            monthly_ct, x="month", y="count", color="outlier_direction",
            barmode="stack",
            color_discrete_map={"고가이상치": "crimson", "저가이상치": "steelblue"},
            labels={"count": "이상치 건수", "month": "연월", "outlier_direction": "유형"},
            title="월별 이상치 거래 건수 (고가/저가 분리)",
            height=320,
        )
        fig_monthly.update_xaxes(tickformat="%Y-%m", dtick="M6", tickangle=-30)
        st.plotly_chart(fig_monthly, width="stretch")

    # 편차 분포 히스토그램
    if "price_deviation_pct" in df.columns:
        fig_hist = px.histogram(
            df, x="price_deviation_pct",
            nbins=60,
            color_discrete_sequence=["#E45756"],
            title="이상치 편차 분포 (시세 대비 %)",
            labels={"price_deviation_pct": "편차 (%)"},
            height=280,
        )
        fig_hist.add_vline(x=25, line_dash="dash", line_color="navy", annotation_text="+25%")
        fig_hist.add_vline(x=-25, line_dash="dash", line_color="navy", annotation_text="-25%")
        st.plotly_chart(fig_hist, width="stretch")

    # 이상치 산점도 (편차% vs 거래가)
    if "price_deviation_pct" in df.columns and "price_per_m2" in df.columns:
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="price_per_m2", y="price_deviation_pct",
            color="outlier_direction" if "outlier_direction" in sample.columns else None,
            color_discrete_map={"고가이상치": "crimson", "저가이상치": "steelblue"},
            hover_data=[c for c in ["apt_name", "dong", "area", "floor", "month",
                                     "ref_price", "ref_month"] if c in sample.columns],
            title="이상치 산점도 (거래가 vs 시세 대비 편차)",
            labels={"price_per_m2": "거래 ㎡당 가격 (만원/㎡)", "price_deviation_pct": "시세 대비 편차 (%)"},
            height=400,
            opacity=0.55,
        )
        fig_scatter.add_hline(y=25,  line_dash="dash", line_color="crimson", annotation_text="+25% 임계")
        fig_scatter.add_hline(y=-25, line_dash="dash", line_color="steelblue", annotation_text="-25% 임계")
        st.plotly_chart(fig_scatter, width="stretch")

    # 케이스북 테이블
    st.markdown("#### 이상치 케이스북 (편차 절댓값 상위 100건)")
    case_cols = [
        "month", "dong", "apt_name", "area", "area_repr", "floor",
        "price", "price_per_m2",
        "ref_month", "ref_price", "price_deviation_pct", "outlier_direction",
    ]
    case_cols = [c for c in case_cols if c in df.columns]
    top_cases = (
        df[case_cols]
        .sort_values("price_deviation_pct", key=abs, ascending=False)
        .head(100)
        .copy()
    )
    for date_col in ("month", "ref_month"):
        if date_col in top_cases.columns:
            top_cases[date_col] = pd.to_datetime(top_cases[date_col]).dt.strftime("%Y-%m")
    if "price_deviation_pct" in top_cases.columns:
        top_cases["price_deviation_pct"] = top_cases["price_deviation_pct"].round(1)
    st.dataframe(top_cases, width="stretch", height=400)

    # 다운로드
    csv = df.drop(columns=["_sggCd"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="이상치 전체 CSV 다운로드",
        data=csv,
        file_name=f"outliers_{selected_code}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 메인 render
# ---------------------------------------------------------------------------

def render() -> None:
    """시장 스냅샷 페이지를 렌더링한다."""
    st.header("A. 데이터 진단 & 시장 스냅샷")
    st.markdown(
        "아파트 매매·전월세 실거래 데이터의 기초 현황을 진단하고, "
        "시장의 흐름을 한눈에 파악합니다."
    )

    # 데이터 로드
    trade_df = load_snapshot_monthly_trade()
    rent_df = load_snapshot_monthly_rent()
    area_mix_df = load_snapshot_area_mix()
    outliers_df = load_snapshot_outliers()

    pipeline_not_run = trade_df.empty and area_mix_df.empty

    if pipeline_not_run:
        st.warning(
            "집계 데이터가 없습니다. 먼저 파이프라인을 실행해주세요.\n\n"
            "```bash\npython pipelines/market_snapshot_pipeline.py\n```"
        )
        return

    # 사이드바 지역 선택
    region_opts = _region_options()
    region_display = list(region_opts.values())
    region_codes = list(region_opts.keys())

    selected_name = st.sidebar.selectbox(
        "지역 선택 (Section A)",
        region_display,
        index=0,
        key="snapshot_region",
    )
    selected_code = region_codes[region_display.index(selected_name)]

    # 기간 범위 슬라이더
    if not trade_df.empty and "month" in trade_df.columns:
        min_date = trade_df["month"].min()
        max_date = trade_df["month"].max()
        min_year = int(min_date.year)
        max_year = int(max_date.year)

        year_range = st.sidebar.slider(
            "조회 연도 범위",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="snapshot_year_range",
        )
        # 기간 필터 적용
        trade_df = trade_df[
            (trade_df["month"].dt.year >= year_range[0])
            & (trade_df["month"].dt.year <= year_range[1])
        ]
        if not rent_df.empty:
            rent_df = rent_df[
                (rent_df["month"].dt.year >= year_range[0])
                & (rent_df["month"].dt.year <= year_range[1])
            ]
        if not area_mix_df.empty:
            area_mix_df = area_mix_df[
                (area_mix_df["month"].dt.year >= year_range[0])
                & (area_mix_df["month"].dt.year <= year_range[1])
            ]

    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "A-1. 월별 거래량·가격",
        "A-2. 면적 믹스",
        "A-3. 이상치 탐지",
    ])

    with tab1:
        _render_a1(trade_df, rent_df, selected_code)

    with tab2:
        _render_a2(area_mix_df, selected_code)

    with tab3:
        _render_a3(outliers_df, selected_code)
