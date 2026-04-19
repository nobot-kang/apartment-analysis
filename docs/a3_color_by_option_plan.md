# A-3 이상치 그래프 — 색상(라벨) 기준 선택 옵션 추가 계획

## 배경

`dashboard/pages/page_00_market_snapshot_diagnostics.py` 의 **A-3. 이상치·오류·비정상 거래 탐지** 섹션에는 두 개의 그래프(월별 이상치 추이 막대, 이상치 산점도)가 색상으로 카테고리를 구분한다. 현재 색상 컬럼(`color_col`, `scatter_color`)은 "판정 사유 필터" 선택값에 암묵적으로 묶여 있다:

- 필터가 `전체` 인 경우 → `판정사유` 로 색칠
- 필터가 특정 사유를 고른 경우 → `outlier_direction` (고가/저가) 으로 색칠

문제: 사용자는 **필터를 `전체` 로 둔 채로도 `outlier_direction` 기준의 고가/저가 분포**를 보고 싶어하는 경우가 있다. 현재 구조에서는 두 축을 독립적으로 선택할 수 없다.

관련 위치:
- `page_00_market_snapshot_diagnostics.py:495` — `color_col` 결정
- `page_00_market_snapshot_diagnostics.py:575` — `scatter_color` 결정

## 목표

"판정 사유 필터" 와 별개로, 그래프의 **색상 기준(라벨 축)을 사용자 선택 옵션으로 분리**한다. 기본값은 기존 동작(필터에 따른 자동 선택)과 호환되도록 하되, 사용자가 명시적으로 `판정사유` 또는 `outlier_direction` 을 고를 수 있게 한다.

## UI 설계

### 위치
"판정 사유 필터" 와 "단지 유형 필터" 셀렉트박스 바로 아래 (`page_00_market_snapshot_diagnostics.py:463~488` 이후).

### 컨트롤
`st.radio` (또는 `st.selectbox`) — 3지 선택:

| 라벨 | 값 | 동작 |
|---|---|---|
| 자동 (필터 연동) | `auto` | 기존 동작: 필터가 `전체` 면 `판정사유`, 아니면 `outlier_direction` |
| 판정사유 | `reason` | 항상 `판정사유` 로 색칠 |
| 고가/저가 (outlier_direction) | `direction` | 항상 `outlier_direction` 으로 색칠 |

```python
color_mode = st.radio(
    "그래프 색상 기준",
    options=["auto", "reason", "direction"],
    format_func={"auto": "자동 (필터 연동)",
                 "reason": "판정사유",
                 "direction": "고가/저가 방향"}.get,
    horizontal=True,
    key="a3_color_mode",
)
```

`horizontal=True` 로 하여 공간을 아끼고, `key` 를 지정하여 세션 상태를 유지한다.

## 로직 변경

### resolver 시그니처

`reason_key` 만으로는 `outlier_direction` 컬럼 누락을 판단할 수 없다. `auto` 모드에서 특정 사유가 선택되면 결과적으로 `outlier_direction` 을 고르게 되므로, **`direction` 모드뿐 아니라 `auto -> direction` 경로에서도 컬럼 부재 시 동일하게 `판정사유` 로 폴백**해야 한다. 따라서 resolver 는 `available_columns` 를 인자로 받는다:

```python
from collections.abc import Iterable

def _resolve_color_col(
    mode: str,
    reason_key: str | None,
    available_columns: Iterable[str],
) -> tuple[str, bool]:
    """(color_col, fell_back_from_direction) 반환."""
    cols = set(available_columns)

    if mode == "reason":
        return "판정사유", False

    wants_direction = (
        mode == "direction"
        or (mode == "auto" and reason_key is not None)
    )
    if wants_direction:
        if "outlier_direction" in cols:
            return "outlier_direction", False
        return "판정사유", True  # old schema 폴백

    # mode == "auto" and reason_key is None
    return "판정사유", False
```

반환 튜플의 두 번째 값은 "폴백이 실제로 발생했는지" 플래그다. 호출부에서 `st.info` 를 띄울지 결정하는 데만 쓰이고, 색상 결정 자체에는 영향을 주지 않는다.

### 호출 위치 — 한 번만 해석

`page_00_market_snapshot_diagnostics.py:495` 와 `:575` 에서 **각각** resolver 를 호출하면 (a) 폴백 경고가 두 번 렌더링되고 (b) 추후 로직이 분기되면 두 그래프의 범례 기준이 어긋날 위험이 있다. 필터가 모두 적용된 직후 한 번만 계산하고 두 그래프에 동일한 값을 넘긴다:

```python
# 필터 적용(판정 사유 + 단지 유형) 직후, 월별 차트 렌더링 이전
resolved_color_col, _direction_fallback = _resolve_color_col(
    color_mode, selected_reason_key, df.columns
)
if _direction_fallback:
    st.info(
        "현재 데이터에 `outlier_direction` 컬럼이 없어 `판정사유` 기준으로 색칠합니다."
    )

# 이후 월별 bar / scatter 양쪽 모두에서 resolved_color_col 사용
color_col = resolved_color_col      # was: line 495
scatter_color = resolved_color_col  # was: line 575
```

기존 `page_00_market_snapshot_diagnostics.py:495` 와 `:575` 의 삼항식은 삭제한다.

## 안전장치

1. **컬럼 존재 확인** — resolver 내부에서 `available_columns` 로 판정하므로 `direction` / `auto -> direction` 양쪽에서 크래시 없이 `판정사유` 로 폴백된다. 경고(`st.info`)는 resolver 가 반환한 플래그를 보고 호출부에서 **한 번만** 띄운다.
2. **필터링 후 카테고리 비어있음** — 예: `direction` 모드에서 한쪽 방향이 필터 결과에 0건일 때 `color_discrete_map` 은 무시되어도 plotly 가 정상 렌더링하므로 별도 처리 불필요. (`color_map` 은 현재 `판정사유` 라벨과 `outlier_direction` 라벨이 모두 섞여 있어 그대로 사용 가능.)
3. **기본값** — `auto` 로 두어 기존 사용자 경험을 보존한다.
4. **단일 해석 지점** — resolver 는 필터 적용 직후에만 호출하고 반환값을 두 Plotly 호출에 재사용한다. 이후 로직을 수정할 때도 두 그래프가 자동으로 동일한 색상 기준을 따른다.

## 검증 항목

- [ ] 필터 `전체` + 색상 `자동` → 기존과 동일하게 `판정사유` 로 색칠
- [ ] 필터 특정 사유 + 색상 `자동` → 기존과 동일하게 `outlier_direction` 으로 색칠
- [ ] 필터 `전체` + 색상 `direction` → 고가/저가 2색 분포가 월별 차트/산점도 모두에 반영 (신규 동작)
- [ ] 필터 특정 사유 + 색상 `reason` → 해당 사유 단일 색 (1개 범례) 로 표시
- [ ] 필터 `전체` + 색상 `direction` + `outlier_direction` 없는 old schema → 크래시 없이 `판정사유` 로 폴백
- [ ] **필터 특정 사유 + 색상 `자동` + `outlier_direction` 없는 old schema → 크래시 없이 `판정사유` 로 폴백** (핵심 회귀 포인트: `auto -> direction` 경로의 폴백)
- [ ] 폴백 발생 시 `st.info` 경고가 페이지당 **한 번만** 표시됨 (월별/산점도 두 그래프에서 중복되지 않음) — *자동 테스트 범위 밖. resolver 는 폴백 플래그만 보장하며, 경고를 1회만 소비하는 것은 렌더링 코드 책임이므로 이 항목은 수동 검증으로 남긴다.*

## 테스트

대시보드 UI 단위 테스트는 현재 비어 있으므로 UI 렌더링 대신 **resolver 순수 함수** 단위 테스트만 추가한다. 모듈 import 시 `streamlit` 부작용을 피하기 위해 `_resolve_color_col` 는 모듈 최상단 헬퍼로 두고 Streamlit 호출과 분리한다.

권장 테스트 케이스 (예: `tests/dashboard/test_a3_color_resolver.py`):

1. `mode="auto"`, `reason_key=None`, 컬럼 있음 → `("판정사유", False)`
2. `mode="auto"`, `reason_key="sanity_error"`, `outlier_direction` 있음 → `("outlier_direction", False)`
3. `mode="auto"`, `reason_key="sanity_error"`, `outlier_direction` 없음 → `("판정사유", True)` *(회귀 방지 핵심)*
4. `mode="direction"`, `outlier_direction` 없음 → `("판정사유", True)`
5. `mode="reason"`, 컬럼 유무 무관 → `("판정사유", False)` *(신규 사용자 선택 분기, 정상 경로)*
6. `mode="direction"`, `outlier_direction` 있음 → `("outlier_direction", False)` *(신규 사용자 선택 분기, 정상 경로)*

5·6 번이 없으면 사용자가 명시적으로 고른 두 분기가 망가져도 테스트가 녹색으로 통과할 수 있다.

> `st.info` 가 페이지당 한 번만 렌더링되는지는 resolver 순수 함수 범위를 벗어난다. 필요 시 "경고를 띄울지"만 결정하는 별도 helper (`_should_warn_direction_fallback(mode, reason_key, cols) -> bool`) 로 분리해 단위 테스트할 수 있지만, 현재 단계에서는 수동 검증으로 충분하다.

## 예상 범위

- 수정 파일: `dashboard/pages/page_00_market_snapshot_diagnostics.py` 1개
- 신규 파일: `tests/dashboard/test_a3_color_resolver.py` (resolver 단위 테스트)
- 수정 라인: 컨트롤 추가 + `_resolve_color_col` 헬퍼 + 공통 해석 지점 + 2개 호출부 교체 (약 25~30줄)
- 파이프라인 영향 없음 (표시 레이어만 변경)
