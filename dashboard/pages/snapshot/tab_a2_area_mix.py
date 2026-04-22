"""A-2: 면적 믹스 변화 & 구성효과 분해."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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
