"""A-3: 이상치·오류·비정상 거래 탐지."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import SEOUL_REGIONS
from dashboard.pages.snapshot.a3_labels import (
    A3_REASON_LABELS,
    A3_REASON_ORDER,
    A3_STRUCTURE_LABELS,
    A3_STRUCTURE_ORDER,
)
from dashboard.pages.snapshot.a3_filters import (
    _ordered_present_keys,
    _prepare_a3_filter_frame,
    _resolve_color_col,
)


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

    if selected_code not in ("ALL", "SEOUL", "GYEONGGI"):
        df = df[df["_sggCd"] == selected_code]
    elif selected_code == "SEOUL":
        df = df[df["_sggCd"].isin(SEOUL_REGIONS.keys())]
    elif selected_code == "GYEONGGI":
        df = df[~df["_sggCd"].isin(SEOUL_REGIONS.keys())]

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
                mp = mp[mp_sgg.isin(SEOUL_REGIONS.keys())]
            elif selected_code == "GYEONGGI":
                mp = mp[~mp_sgg.isin(SEOUL_REGIONS.keys())]
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
