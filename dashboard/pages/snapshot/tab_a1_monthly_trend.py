"""A-1: 월별 거래량·중위 ㎡당 가격·분산 추이."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
