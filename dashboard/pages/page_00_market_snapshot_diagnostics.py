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
    load_snapshot_complex_market_price,
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


A3_REASON_LABELS: dict[str, str] = {
    "unsupported_jump": "지지 없는 단발 점프",
    "sanity_error": "입력/단위 오류",
    "trend_month_robust_band": "추세월 내부 스파이크",
    "abs_deviation": "절대 금액 이탈 (±3억)",
    "legacy_band_outlier": "legacy 밴드 이상치",
    "uncategorized": "미분류",
}

A3_REASON_ORDER = [
    "unsupported_jump",
    "sanity_error",
    "trend_month_robust_band",
    "abs_deviation",
    "legacy_band_outlier",
    "uncategorized",
]

A3_STRUCTURE_LABELS: dict[str, str] = {
    "normal": "일반 단지",
    "leader_or_isolated": "대장/나홀로 단지",
    "legacy_unknown": "미분류 (legacy 데이터)",
    "unknown": "미분류",
}

A3_STRUCTURE_ORDER = [
    "normal",
    "leader_or_isolated",
    "legacy_unknown",
    "unknown",
]


def _resolve_color_col(
    mode: str,
    reason_key: str | None,
    available_columns: "Iterable[str]",
) -> tuple[str, bool]:
    """그래프 색상 컬럼과 폴백 여부를 반환한다.

    Returns:
        (color_col, fell_back_from_direction)
        fell_back_from_direction=True 이면 caller 가 st.info 경고를 1회 표시해야 한다.
    """
    from collections.abc import Iterable  # noqa: PLC0415 — lazy import to keep module-level clean

    cols = set(available_columns)

    if mode == "reason":
        return "판정사유", False

    wants_direction = mode == "direction" or (mode == "auto" and reason_key is not None)
    if wants_direction:
        if "outlier_direction" in cols:
            return "outlier_direction", False
        return "판정사유", True  # old schema 폴백

    # mode == "auto" and reason_key is None
    return "판정사유", False


def _ordered_present_keys(series: pd.Series, preferred_order: list[str]) -> list[str]:
    """선호 순서를 우선한 뒤 나머지 값을 정렬해 반환한다."""
    values = [str(v) for v in series.dropna().unique().tolist() if str(v).strip()]
    present = set(values)
    ordered = [key for key in preferred_order if key in present]
    extras = sorted(present - set(preferred_order))
    return ordered + extras


def _prepare_a3_filter_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool]:
    """A-3 필터/표시용 레이블 컬럼을 준비한다."""
    prepared = df.copy()

    has_reason_schema = (
        "outlier_reason" in prepared.columns
        and prepared["outlier_reason"].dropna().astype(str).str.strip().ne("").any()
    )
    if has_reason_schema:
        reason_key = prepared["outlier_reason"].fillna("uncategorized").astype(str)
        reason_key = reason_key.where(reason_key.str.strip().ne(""), "uncategorized")
    elif "reference_type" in prepared.columns and prepared["reference_type"].notna().any():
        reason_key = (
            prepared["reference_type"]
            .map(
                {
                    "moving_average_band": "legacy_band_outlier",
                    "trend_month_robust_band": "trend_month_robust_band",
                    "sanity_error": "sanity_error",
                }
            )
            .fillna("legacy_band_outlier")
            .astype(str)
        )
    else:
        reason_key = pd.Series("uncategorized", index=prepared.index, dtype="object")
    prepared["_reason_filter_key"] = reason_key
    prepared["판정사유"] = prepared["_reason_filter_key"].map(A3_REASON_LABELS).fillna(prepared["_reason_filter_key"])

    has_structure_schema = (
        "structure_type" in prepared.columns
        and prepared["structure_type"].dropna().astype(str).str.strip().ne("").any()
    )
    if has_structure_schema:
        structure_key = prepared["structure_type"].fillna("unknown").astype(str)
        structure_key = structure_key.where(structure_key.str.strip().ne(""), "unknown")
    else:
        structure_key = pd.Series("legacy_unknown", index=prepared.index, dtype="object")
    prepared["_structure_filter_key"] = structure_key
    prepared["단지유형"] = prepared["_structure_filter_key"].map(A3_STRUCTURE_LABELS).fillna(prepared["_structure_filter_key"])

    return prepared, has_reason_schema, has_structure_schema


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

def _render_a3(
    outliers_df: pd.DataFrame,
    selected_code: str,
    market_price_df: pd.DataFrame | None = None,
) -> None:
    st.subheader("A-3. 이상치·오류·비정상 거래 탐지")
    st.caption(
        "탐지 기준 (v2): 코호트(sggCd × 면적대) 시세 경로 + 단지 고유 spread를 합산한 기준값 대비 "
        "dynamic band(기본 floor ±18%, 희소 그룹 ±23%, 대장/나홀로 ±28% + sigma 확장) 이탈 거래 중, "
        "전후 2~6개월 내 지지 거래가 없는 고립 스파이크만 제거합니다. "
        "추세월 내부 스파이크는 별도 robust band를 사용하며, 노후 단지(築20년+) 고가 거래는 ±5천만원·12% 이내 완화합니다. "
        "| 사전 제외: 1층 거래"
    )

    if outliers_df.empty:
        st.info("이상치 데이터가 없습니다. 파이프라인을 먼저 실행해주세요.")
        st.code("uv run python pipelines/market_snapshot_pipeline.py", language="bash")
        return

    # 지역 필터
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

    df, has_reason_schema, has_structure_schema = _prepare_a3_filter_frame(df)
    if not has_reason_schema or not has_structure_schema:
        missing = []
        if not has_reason_schema:
            missing.append("판정 사유")
        if not has_structure_schema:
            missing.append("단지 유형")
        st.info(
            "현재 `snapshot_outliers.parquet` 에 v2 진단 컬럼이 일부 없어 "
            f"`{', '.join(missing)}` 필터는 fallback 카테고리로 표시됩니다. "
            "최신 파이프라인을 다시 실행하면 세부 분류가 채워집니다."
        )

    # KPI
    high_cnt = (df["outlier_direction"] == "고가이상치").sum() if "outlier_direction" in df.columns else 0
    low_cnt  = (df["outlier_direction"] == "저가이상치").sum() if "outlier_direction" in df.columns else 0
    reno_cnt = 0
    if market_price_df is not None and not market_price_df.empty and "renovation_buffer_count" in market_price_df.columns:
        mp = market_price_df
        if "aptSeq" in mp.columns:
            mp_sgg = mp["aptSeq"].astype(str).str.split("-").str[0]
            if selected_code not in ("ALL", "SEOUL", "GYEONGGI"):
                mp = mp[mp_sgg == selected_code]
            elif selected_code == "SEOUL":
                mp = mp[mp_sgg.isin(_SEOUL_REGIONS.keys())]
            elif selected_code == "GYEONGGI":
                mp = mp[~mp_sgg.isin(_SEOUL_REGIONS.keys())]
        reno_cnt = int(mp["renovation_buffer_count"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 이상치 건수", f"{len(df):,}")
    col2.metric("고가 이상치 (↑)", f"{high_cnt:,}")
    col3.metric("저가 이상치 (↓)", f"{low_cnt:,}")
    col4.metric("노후단지 완화 적용", f"{reno_cnt:,}", help="renovation_buffer 로 outlier 해제된 거래 수 (snapshot_complex_market_price 기준)")

    # 판정 사유 필터
    reason_pairs = [("전체", None)] + [
        (A3_REASON_LABELS.get(key, key), key)
        for key in _ordered_present_keys(df["_reason_filter_key"], A3_REASON_ORDER)
    ]
    selected_reason_label = st.selectbox(
        "판정 사유 필터",
        [label for label, _ in reason_pairs],
        key="a3_reason_filter",
    )
    selected_reason_key = next(key for label, key in reason_pairs if label == selected_reason_label)
    if selected_reason_key is not None:
        df = df[df["_reason_filter_key"] == selected_reason_key]

    structure_pairs = [("전체", None)] + [
        (A3_STRUCTURE_LABELS.get(key, key), key)
        for key in _ordered_present_keys(df["_structure_filter_key"], A3_STRUCTURE_ORDER)
    ]
    selected_structure_label = st.selectbox(
        "단지 유형 필터",
        [label for label, _ in structure_pairs],
        key="a3_structure_filter",
    )
    selected_structure_key = next(key for label, key in structure_pairs if label == selected_structure_label)
    if selected_structure_key is not None:
        df = df[df["_structure_filter_key"] == selected_structure_key]

    if df.empty:
        st.warning("필터 조건에 해당하는 이상치가 없습니다.")
        return

    # 그래프 색상 기준 선택
    color_mode = st.radio(
        "그래프 색상 기준",
        options=["auto", "reason", "direction"],
        format_func={"auto": "자동 (필터 연동)", "reason": "판정사유", "direction": "고가/저가 방향"}.get,
        horizontal=True,
        key="a3_color_mode",
    )
    resolved_color_col, _direction_fallback = _resolve_color_col(color_mode, selected_reason_key, df.columns)
    if _direction_fallback:
        st.info("현재 데이터에 `outlier_direction` 컬럼이 없어 `판정사유` 기준으로 색칠합니다.")

    # 월별 이상치 건수 추이 (고가/저가 + 판정사유 분리)
    color_col = resolved_color_col
    if color_col in df.columns:
        monthly_ct = (
            df.groupby(["month", color_col])
            .size()
            .reset_index(name="count")
            .sort_values("month")
        )
        color_map = {
            "고가이상치": "crimson", "저가이상치": "steelblue",
            "지지 없는 단발 점프": "darkorange",
            "입력/단위 오류": "red",
            "추세월 내부 스파이크": "purple",
            "절대 금액 이탈 (±3억)": "teal",
            "legacy 밴드 이상치": "indianred",
            "미분류": "gray",
        }
        fig_monthly = px.bar(
            monthly_ct, x="month", y="count", color=color_col,
            barmode="stack",
            color_discrete_map=color_map,
            labels={"count": "이상치 건수", "month": "연월"},
            title="월별 이상치 거래 건수",
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
            title="이상치 편차 분포 (기준 시세 대비 %)",
            labels={"price_deviation_pct": "편차 (%)"},
            height=280,
        )
        fig_hist.add_vline(x=18, line_dash="dash", line_color="navy", annotation_text="+18% 기본 floor")
        fig_hist.add_vline(x=-18, line_dash="dash", line_color="navy", annotation_text="-18% 기본 floor")
        band_series = (
            pd.to_numeric(df["band_width_pct"], errors="coerce").dropna()
            if "band_width_pct" in df.columns
            else pd.Series(dtype=float)
        )
        if not band_series.empty:
            band_p50 = float(band_series.median())
            if band_p50 > 18.0:
                fig_hist.add_vline(
                    x=band_p50,
                    line_dash="dot",
                    line_color="darkorange",
                    annotation_text=f"+{band_p50:.1f}% 실제 중앙 band",
                )
                fig_hist.add_vline(
                    x=-band_p50,
                    line_dash="dot",
                    line_color="darkorange",
                    annotation_text=f"-{band_p50:.1f}% 실제 중앙 band",
                )
        st.plotly_chart(fig_hist, width="stretch")
        if not band_series.empty:
            band_p50 = float(band_series.median())
            band_p90 = float(band_series.quantile(0.9))
            band_max = float(band_series.max())
            st.caption(
                f"`±18%`는 기본 floor 가이드라인입니다. 실제 cutoff는 각 거래의 `band_width_pct`이며 "
                f"현재 결과 기준 중앙값은 `±{band_p50:.1f}%`, 90% 분위는 `±{band_p90:.1f}%`, 최대는 `±{band_max:.1f}%`입니다. "
                "희소 그룹은 `±23%`, 대장/나홀로는 `±28%` floor가 적용될 수 있고, "
                "`trend_month_robust_band` 행은 별도 robust band를 사용하므로 분포가 더 넓거나 좁게 보이는 것이 정상입니다."
            )

    # 이상치 산점도 (편차% vs 거래가)
    if "price_deviation_pct" in df.columns and "price_per_m2" in df.columns:
        hover_cols = [c for c in [
            "apt_name", "dong", "area", "floor", "month",
            "ref_price", "reference_type", "판정사유",
            "단지유형", "cohort_level", "spread_shrunk",
            "deviation_total_krw", "renovation_buffer_applied",
        ] if c in df.columns]
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="price_per_m2", y="price_deviation_pct",
            color=resolved_color_col,
            color_discrete_map={
                "고가이상치": "crimson", "저가이상치": "steelblue",
                "지지 없는 단발 점프": "darkorange",
                "입력/단위 오류": "red",
                "추세월 내부 스파이크": "purple",
                "legacy 밴드 이상치": "indianred",
                "미분류": "gray",
            },
            hover_data=hover_cols,
            title="이상치 산점도 (거래가 vs 기준 시세 대비 편차)",
            labels={"price_per_m2": "거래 ㎡당 가격 (만원/㎡)", "price_deviation_pct": "편차 (%)"},
            height=400,
            opacity=0.55,
        )
        fig_scatter.add_hline(y=18,  line_dash="dash", line_color="crimson", annotation_text="+18% floor")
        fig_scatter.add_hline(y=-18, line_dash="dash", line_color="steelblue", annotation_text="-18% floor")
        st.plotly_chart(fig_scatter, width="stretch")

    # 케이스북 테이블
    st.markdown("#### 이상치 케이스북 (편차 절댓값 상위 100건)")
    case_cols = [
        "month", "dong", "apt_name", "area", "area_repr", "floor", "age",
        "price", "price_per_m2",
        "ref_price", "band_width_pct", "price_deviation_pct", "deviation_total_krw",
        "outlier_direction", "판정사유", "reference_type",
        "단지유형", "cohort_level", "spread_shrunk",
        "renovation_buffer_applied",
    ]
    case_cols = [c for c in case_cols if c in df.columns]
    top_cases = (
        df[case_cols]
        .sort_values("price_deviation_pct", key=abs, ascending=False)
        .head(100)
        .copy()
    )
    if "month" in top_cases.columns:
        top_cases["month"] = pd.to_datetime(top_cases["month"]).dt.strftime("%Y-%m")
    for pct_col in ("band_width_pct", "price_deviation_pct"):
        if pct_col in top_cases.columns:
            top_cases[pct_col] = top_cases[pct_col].round(1)
    if "spread_shrunk" in top_cases.columns:
        top_cases["spread_shrunk"] = top_cases["spread_shrunk"].round(3)
    if "deviation_total_krw" in top_cases.columns:
        top_cases["deviation_total_krw"] = top_cases["deviation_total_krw"].round(0).astype("Int64")
        top_cases = top_cases.rename(columns={"deviation_total_krw": "편차(만원)"})
    st.dataframe(top_cases, width="stretch", height=400)

    # 다운로드
    csv = df.drop(
        columns=["_sggCd", "_reason_filter_key", "_structure_filter_key"],
        errors="ignore",
    ).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="이상치 전체 CSV 다운로드",
        data=csv,
        file_name=f"outliers_{selected_code}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 메인 render
# ---------------------------------------------------------------------------

def render_snapshot() -> None:
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
    market_price_df = load_snapshot_complex_market_price()

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
        _render_a3(outliers_df, selected_code, market_price_df)
