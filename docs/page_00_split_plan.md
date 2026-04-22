# page_00_market_snapshot_diagnostics 분리 계획 (v8)

## 배경 / 목표

- `dashboard/pages/page_00_market_snapshot_diagnostics.py` 는 **767 라인** 단일 파일로 A-1, A-2, A-3 섹션 렌더링, 공통 헬퍼, 이상치 상수/라벨/전처리가 한 곳에 뭉쳐 있다. 편집 시 무관한 섹션까지 통째로 로드해야 한다.
- 탭(A-1 / A-2 / A-3)과 A-3 전용 순수 헬퍼를 별도 모듈로 분리해 **토큰 효율**을 높이고, 차후 각 탭 단위의 수정·리뷰·테스트 범위를 좁힌다.
- 페이지 엔트리포인트 `render_snapshot` 은 `dashboard.pages.page_00_market_snapshot_diagnostics` 경로에 그대로 유지해 `dashboard/app.py:20` 의 동적 import 와 `streamlit run` 호환성을 깨지 않는다.
- **이번 PR 은 순수한 파일 이동/분리** 에 집중한다. 동작(렌더 결과, 필터 동작, CSV 출력, KPI 계산)은 불변이며, 개선 아이디어는 전부 후속 PR 로 분리한다.

## 저장소 내 기존 소비자 (조사 결과)

| 경로 | 의존 심볼 | 처리 방침 |
| --- | --- | --- |
| `dashboard/app.py:20` | `render_snapshot` | 엔트리포인트 경로 보존 → 변경 불필요 |
| `tests/dashboard/test_a3_color_resolver.py:7` | `_resolve_color_col` (엔트리포인트 모듈에서 직접 import) | ① 엔트리포인트에서 `_resolve_color_col` **한 심볼만** 임시 re-export 해 현 테스트를 즉시 통과시키고, ② 같은 PR 에서 테스트를 신규 모듈 경로(`dashboard.pages.snapshot.a3_filters`)로 이전 |

> `rg "page_00_market_snapshot_diagnostics|_resolve_color_col|_ordered_present_keys|_prepare_a3_filter_frame|_render_a[123]|A3_REASON|A3_STRUCTURE"` 결과 상기 두 파일 외 내부 소비자는 없다. 문서/plan md 는 설명 목적이며 import 하지 않는다. 따라서 **하위 호환 re-export 범위는 `_resolve_color_col` 로 한정**한다. `_ordered_present_keys` · `_prepare_a3_filter_frame` 는 기존에 외부 소비자가 없으므로 re-export 하지 않고 처음부터 `dashboard.pages.snapshot.a3_filters` 만 정식 경로로 둔다.

## 현재 구조 요약 (source: `dashboard/pages/page_00_market_snapshot_diagnostics.py`, 767 lines)

| 구간 | 라인 | 책임 |
| --- | --- | --- |
| docstring + import + `sys.path` bootstrap | 1–29 (bootstrap 은 18–20) | 모듈 import 및 `_project_root` 를 `sys.path` 에 주입 |
| `_region_options`, `_plotly_line` | 36–65 | 공통 헬퍼 |
| `A3_REASON_*`, `A3_STRUCTURE_*` 상수 | 68–98 | A-3 전용 라벨/정렬 |
| `_resolve_color_col`, `_ordered_present_keys`, `_prepare_a3_filter_frame` | 101–179 | A-3 전용 순수 헬퍼 |
| `_render_a1` | 186–288 | A-1 렌더 |
| `_render_a2` | 295–411 | A-2 렌더 |
| `_render_a3` | 418–679 | A-3 렌더 (최대 밀집 구간) |
| `render_snapshot` | 686–767 | 데이터 로드, 지역/연도 필터, 탭 구성 |

## 분리 후 구조

```
dashboard/pages/
    page_00_market_snapshot_diagnostics.py   # 얇은 엔트리포인트 (render_snapshot + 하위 호환 re-export)
    snapshot/                                # 신규 서브패키지
        __init__.py                          # 비워둔다 (namespace 역할만; render_snapshot 은 기존 경로 유지)
        _common.py                           # _region_options, _plotly_line
        tab_a1_monthly_trend.py              # _render_a1
        tab_a2_area_mix.py                   # _render_a2
        tab_a3_outliers.py                   # _render_a3
        a3_labels.py                         # A3_REASON_* / A3_STRUCTURE_* 상수
        a3_filters.py                        # _resolve_color_col, _ordered_present_keys, _prepare_a3_filter_frame (순수 헬퍼)
```

### 각 파일 역할과 soft target 라인수

> 숫자는 **가이드**일 뿐 hard requirement 가 아니다. 가독성/응집도가 우선이며, 초과해도 설계상 이유가 정당하면 그대로 둔다.

| 파일 | 포함 | soft target |
| --- | --- | --- |
| `page_00_market_snapshot_diagnostics.py` | `render_snapshot` + 하위 호환 re-export + bootstrap | ~100 lines |
| `snapshot/__init__.py` | 빈 파일(또는 한줄 docstring) | ≤ 5 lines |
| `snapshot/_common.py` | `_region_options`, `_plotly_line` (현재 저장소에서 호출자 없음 — `rg _plotly_line` 으로 확인됨. 본 PR 은 **unused 이지만 그대로 이동**하고, 삭제/정리는 후속 cleanup PR 로 분리) | ~40 lines |
| `snapshot/tab_a1_monthly_trend.py` | `_render_a1` | ~120 lines |
| `snapshot/tab_a2_area_mix.py` | `_render_a2` | ~130 lines |
| `snapshot/tab_a3_outliers.py` | `_render_a3` | 250 lines 내외 (압축 금지) |
| `snapshot/a3_labels.py` | 상수 4종 | ~40 lines |
| `snapshot/a3_filters.py` | 순수 헬퍼 3종 | ~80 lines |

### Import 원칙

각 모듈의 import 허용 범위는 아래 표에 **whitelist 형태**로 고정한다. 리뷰 시 whitelist 밖 import 가 생기면 차단한다.

| 모듈 | 허용 import | 금지 import | 비고 |
| --- | --- | --- | --- |
| `snapshot/a3_labels.py` | (없음) | 전부 | 상수 전용. 순수 파이썬 리터럴만 포함 |
| `snapshot/a3_filters.py` | `pandas`, `collections.abc`, `snapshot.a3_labels` | `streamlit`, `plotly`, `dashboard.data_loader`, `config.*` | 순수 헬퍼. hermetic 테스트 대상 |
| `snapshot/_common.py` | `pandas`, `plotly` (`plotly.express`, `plotly.graph_objects`), `config.settings` | `streamlit`, `dashboard.data_loader` | `_region_options` 이 `SEOUL_REGIONS`/`GYEONGGI_REGIONS` 를 참조하므로 `config.settings` 는 필수. `streamlit` 은 현 구현상 불필요하므로 금지 |
| `snapshot/tab_a1_monthly_trend.py` | `pandas`, `plotly`, `streamlit`, `snapshot._common` | `dashboard.data_loader` | 데이터는 인자로만 받음 |
| `snapshot/tab_a2_area_mix.py` | `pandas`, `plotly`, `streamlit`, `snapshot._common` | `dashboard.data_loader` | 동일 |
| `snapshot/tab_a3_outliers.py` | `pandas`, `plotly`, `streamlit`, `snapshot.{a3_labels,a3_filters,_common}`, `config.settings` (현 `_render_a3` 가 `SEOUL_REGIONS` 를 사용) | `dashboard.data_loader` | 동일 |
| `page_00_market_snapshot_diagnostics.py` (엔트리포인트) | 위 서브모듈 + `dashboard.data_loader` + `streamlit` | — | **이 refactor 범위에서 `dashboard/pages/snapshot/` 아래 어느 파일도 `dashboard.data_loader` 를 import 하지 않도록 하는 유일한 loader 진입점** (다른 `page_0X_*.py` 들은 각자 loader 를 직접 import 하지만 그건 본 refactor 범위 밖). "허용" 은 drift 방어용 상한이며, **예상 최종 import 집합은 최소화** — 구체적으로 `sys`/`pathlib`(bootstrap), `streamlit`, `dashboard.data_loader`, 서브모듈에서 `_render_a1/_a2/_a3` + `_region_options` (`render_snapshot` 이 직접 호출) + re-export 용 `_resolve_color_col` 만. `pandas`/`plotly` 는 서브모듈이 들고가므로 엔트리포인트에서 제거. **`config.settings` 도 `_region_options` 가 `_common.py` 로, `SEOUL_REGIONS` 사용이 `tab_a3_outliers.py` 로 이동하면서 엔트리포인트에서 제거**한다 |

핵심 원칙:

- **본 refactor 범위(`dashboard/pages/snapshot/*` + `page_00_market_snapshot_diagnostics.py`) 내에서 `dashboard.data_loader` 의 소비자는 오직 엔트리포인트(`render_snapshot`)**. 탭 모듈은 DataFrame 을 **인자로만** 받아 현행 `_render_a1/a2/a3` 시그니처를 그대로 유지한다. (참고: `page_01_overview.py` 등 다른 `page_0X_*.py` 들도 각자 loader 를 import 하지만 그건 본 PR 범위 밖이며, 이번 PR 은 repo-wide uniqueness 를 주장하지 않는다.)
- **`a3_labels.py` / `a3_filters.py`** 는 streamlit 런타임·로컬 parquet 없이 import 가능해야 한다 (hermetic 테스트 전제).
- **`_common.py`** 는 추후 `st.*` 가 필요한 헬퍼가 생기면 `_common.py` 가 아닌 별도 모듈(예: `_common_ui.py`) 로 분리해 purity boundary 를 유지한다.
- `sys.path` 주입은 엔트리포인트에서 선행되므로 서브모듈에서 다시 수행하지 않는다.
- 의존 방향: `tab_a3_outliers` → {`a3_filters`, `a3_labels`, `_common`}, `a3_filters` → `a3_labels`. 역방향 의존 금지 → 순환 import 차단.

## 작업 단계

1. **서브패키지 스캐폴딩**: `dashboard/pages/snapshot/__init__.py` 를 빈 파일로 추가.
2. **공통 헬퍼 이동** → `snapshot/_common.py`: `_region_options`, `_plotly_line` 을 시그니처/타입힌트 그대로 이동.
3. **A-3 상수 이동** → `snapshot/a3_labels.py`: `A3_REASON_LABELS`, `A3_REASON_ORDER`, `A3_STRUCTURE_LABELS`, `A3_STRUCTURE_ORDER` 이동.
4. **A-3 순수 헬퍼 이동** → `snapshot/a3_filters.py`: `_resolve_color_col`, `_ordered_present_keys`, `_prepare_a3_filter_frame` 이동. `a3_labels` 에서 상수 import. `streamlit` import 하지 않음을 확인.
5. **탭 렌더러 이동**:
   - `snapshot/tab_a1_monthly_trend.py` ← `_render_a1` (함수명 유지).
   - `snapshot/tab_a2_area_mix.py` ← `_render_a2`.
   - `snapshot/tab_a3_outliers.py` ← `_render_a3`, 필요한 상수/헬퍼만 `a3_labels`, `a3_filters` 에서 import.
6. **엔트리포인트 슬림화**: `page_00_market_snapshot_diagnostics.py` 를 다음 순서로 정리한다.
   - docstring.
   - 기존 `sys.path` bootstrap (현재 18–20 라인) 을 **그대로 유지**한다. 이유: 이 파일은 Streamlit 의 page import 대상이자 `dashboard.app.py:20` 의 동적 import 대상이므로, 엔트리포인트 자신이 `from dashboard.data_loader import ...` 같은 절대 패키지 경로를 Streamlit page-loading 환경과 (`python -c "import dashboard.pages..."` 류) 직접 모듈 import 환경 양쪽에서 resolve 할 수 있어야 한다. 하위 서브모듈(`dashboard/pages/snapshot/*`) 의 절대 import resolution 도 같은 `sys.path` 주입에 의존한다.
   - 사용처만 절대 import:
     * `from dashboard.pages.snapshot._common import _region_options` — `render_snapshot` 이 본문에서 이 helper 를 호출하므로 필수. 엔트리포인트 내부의 기존 `_region_options` 정의는 제거한다.
     * `from dashboard.pages.snapshot.tab_a1_monthly_trend import _render_a1`
     * `from dashboard.pages.snapshot.tab_a2_area_mix import _render_a2`
     * `from dashboard.pages.snapshot.tab_a3_outliers import _render_a3`
   - `render_snapshot` 함수 본문은 그대로 유지 (호출부 변경 없음 — `_region_options()` / `_render_a1/a2/a3(...)` 호출이 전부 import 로 resolve).
   - **하위 호환 re-export**: `_resolve_color_col` **만** `from dashboard.pages.snapshot.a3_filters import _resolve_color_col` 로 re-export. `__all__` 에 포함. `_ordered_present_keys`, `_prepare_a3_filter_frame` 은 기존 소비자가 없으므로 re-export 하지 않는다. 제거 시점은 코드 주석으로 명시: `# TODO(cleanup): remove once no repo-wide usage of _resolve_color_col via this module — check (a) direct import "from dashboard.pages.page_00_market_snapshot_diagnostics import _resolve_color_col" (별칭 포함), (b) attribute access "page_00_market_snapshot_diagnostics._resolve_color_col" / "page_00_market_snapshot_diagnostics.sth.sth" 양쪽 모두 0건인지.`
7. **테스트 이전 및 추가** (같은 PR, 모두 **hermetic** — 실제 parquet/Streamlit 런타임 의존 금지. 입력은 코드 내 `pd.DataFrame(...)` 리터럴로만 생성):
   - `tests/dashboard/test_a3_color_resolver.py` 의 import 경로를 `dashboard.pages.snapshot.a3_filters` 로 갱신. 기존 parametrize 8케이스는 그대로 둬 behavior regression 을 잡는다.
   - `_ordered_present_keys`, `_prepare_a3_filter_frame` 단위 테스트를 신규 `tests/dashboard/test_a3_filter_prep.py` 로 추가. 최소 케이스 (모두 inline DataFrame fixture 사용):
     * `_ordered_present_keys`: preferred order 우선 + extras 정렬.
     * `_prepare_a3_filter_frame`: ①신규 스키마(`outlier_reason`/`structure_type` 채워진 DataFrame), ②legacy 스키마(`reference_type` 만 존재하는 DataFrame, 값은 `moving_average_band` / `trend_month_robust_band` / `sanity_error` 3종), ③두 컬럼 모두 없는 폴백 DataFrame → 각 케이스에서 `has_reason_schema` / `has_structure_schema` 플래그와 `판정사유` / `단지유형` 매핑값을 assert.
   - Entry re-export smoke test: `from dashboard.pages.page_00_market_snapshot_diagnostics import _resolve_color_col` 가 `dashboard.pages.snapshot.a3_filters._resolve_color_col` 과 동일 객체임을 `is` 로 검증.
8. **검증**:
   - `uv run pytest tests/dashboard/` → 기존 resolver 테스트 + 신규 테스트 통과.
   - `uv run python -c "from dashboard.pages.page_00_market_snapshot_diagnostics import render_snapshot; print(render_snapshot)"` → import smoke.
   - `uv run streamlit run streamlit_app.py` → 수동 검증 체크리스트(아래) 전 항목 통과.

## 회귀 확인 체크리스트 (자동 + 수동)

**자동** (전부 hermetic — 로컬 parquet 없이 CI 에서 그대로 수행 가능)
- [ ] `uv run pytest -q` 전체 통과 (현 baseline 은 30 케이스 green 이며, 본 PR 에서는 최소 +5 케이스 추가 예상).
- [ ] `uv run pytest tests/dashboard/test_a3_color_resolver.py` — 경로 변경 후에도 8개 parametrize 케이스 전부 통과.
- [ ] `uv run pytest tests/dashboard/test_a3_filter_prep.py` — 신규 케이스 통과 (inline DataFrame fixture).
- [ ] Entry re-export 동일성 smoke (`is` 비교).
- [ ] Positive smoke: `uv run python -c "from dashboard.pages.page_00_market_snapshot_diagnostics import render_snapshot, _resolve_color_col"` 오류 없음.
- [ ] Negative smoke (정확히 하나의 기대 동작만 고정 — `ImportError`): `uv run python -c "from dashboard.pages.page_00_market_snapshot_diagnostics import _ordered_present_keys"` 와 `uv run python -c "from dashboard.pages.page_00_market_snapshot_diagnostics import _prepare_a3_filter_frame"` 가 모두 `ImportError` 로 실패한다. (`from ... import NAME` 구문은 심볼 부재 시 `ImportError` 를 raise 하므로 `AttributeError` 는 기대 동작에 포함하지 않는다.)

**수동 (Streamlit — 구현 후 유저가 직접 수행)**

> 레거시 스키마(`outlier_direction` 누락, `outlier_reason`/`structure_type` 공백) 에 대한 fallback 동작은 **자동 테스트(`test_a3_filter_prep.py`, `test_a3_color_resolver.py`) 에서 hermetic fixture 로 이미 커버**한다. 따라서 Streamlit 수동 QA 는 **현재 운영 parquet 하나만** 사용하면 되고, 별도 레거시 parquet 을 준비/선택할 필요가 없다. 이로 인해 rollout signoff 의 재현성 문제는 해소된다.

현재 parquet (`uv run streamlit run streamlit_app.py`) 기준 확인 항목:
- [ ] 사이드바 지역 선택(`snapshot_region`) 및 연도 범위(`snapshot_year_range`) 이 변경 후에도 페이지 재진입 시 유지.
- [ ] A-1 탭: KPI 3개, 거래량 바 차트(매매/전세/월세), 중위가 + IQR + 3/12개월 이동평균, 표준편차 차트 렌더.
- [ ] A-2 탭: stacked area(면적 비중), 면적 구간별 가격 라인, 구성효과 실제/고정가중 비교, 구성효과 bar, 요약 `st.info` 출력.
- [ ] A-3 탭: KPI 4개, 판정사유/단지유형 필터(`a3_reason_filter`, `a3_structure_filter`) 및 색상모드(`a3_color_mode`) session_state key 유지. 월별 bar, 편차 분포 히스토그램 + `band_width_pct` 중앙 band 라인, 산점도, 케이스북 테이블 상위 100건, CSV 다운로드(UTF-8-SIG) 가 이전과 동일하게 노출.
- [ ] CSV 다운로드 파일의 컬럼 수/이름이 이전과 동일한지 (내부 키 컬럼 `_sggCd`, `_reason_filter_key`, `_structure_filter_key` 제거 유지).

## 유지해야 할 불변 조건

- `dashboard.pages.page_00_market_snapshot_diagnostics.render_snapshot` 경로·시그니처·동작 불변. `dashboard/app.py:20` 의 동적 import 가 그대로 동작한다.
- **기존 테스트가 의존하는 helper import 경로는 같은 PR 에서 함께 정리한다.** 구체적으로:
  * 이전 경로(`dashboard.pages.page_00_market_snapshot_diagnostics._resolve_color_col`) **한 심볼만** 이번 PR 에서 re-export 로 살린다. 내부 소비자가 없는 `_ordered_present_keys` / `_prepare_a3_filter_frame` 은 re-export 하지 않는다 — 임시 호환 표면을 필요 이상으로 넓히지 않는다.
  * 모든 A-3 helper 테스트는 신규 경로(`dashboard.pages.snapshot.a3_filters`) 기준으로 작성/이전한다.
  * re-export 제거 조건은 **저장소 전체에서 해당 심볼의 엔트리포인트 경유 사용이 0건** 일 때. grep 은 heuristic 이므로 단일 패턴이 아닌 아래 두 rg + 전체 pytest 조합으로 확인한다 (동적 import / 복잡한 alias 는 완전 탐지 보장 없음 — 그래서 `pytest` 를 최종 안전망으로 함께 둔다):
    1. **Direct import** (별칭 포함): `rg -n "from dashboard\.pages\.page_00_market_snapshot_diagnostics import\b[^#\n]*\b_resolve_color_col\b"` (multi-line import 가 있으면 수동 검수).
    2. **Attribute access**: `rg -n "page_00_market_snapshot_diagnostics.*\._resolve_color_col"` — 모듈을 `import ... as m` 으로 alias 하고 `m._resolve_color_col` 로 접근하는 경우를 잡기 위함. alias 가 의심되면 `rg -n "import dashboard\.pages\.page_00_market_snapshot_diagnostics"` 도 함께 확인.
    3. **Safety net**: 제거 PR 에서 위 두 rg 가 0건임을 확인한 뒤 `uv run pytest -q` 가 green 인지 함께 검증. pytest 가 red 이면 rg 로 잡히지 않은 소비자가 존재한다는 신호이므로 제거 보류.

    rg 2건 + pytest 통과가 모두 충족되어야 제거. 조건 충족 시 별도 cleanup PR 에서 제거.
- Streamlit `session_state` key: `a3_reason_filter`, `a3_structure_filter`, `a3_color_mode`, `snapshot_region`, `snapshot_year_range` — 이름/타입 불변.
- 탭 순서/제목/캡션 문구/`st.info`·`st.warning` 본문 변경 금지.
- A-3 색상 매핑 딕셔너리(`color_map`) 는 현재 `_render_a3` 내부에 중복 정의되어 있다. **이번 PR 에서는 건드리지 않는다.** 중복 제거는 후속 cleanup PR 로 분리.
- CSV 다운로드 인코딩 `utf-8-sig`, 파일명 포맷 `outliers_{selected_code}.csv` 불변.

## 범위 외 (이번 PR 에서 하지 않음)

- A-3 판정 로직, 라벨 문구, KPI 계산식, band 경계값 변경.
- 다른 `page_0X_*.py` 분리. 본 PR 결과를 바탕으로 후속 PR 로 동일 패턴 적용 가능 여부만 평가.
- `color_map` 중복 제거, **`_plotly_line` (현재-unused) 삭제 또는 실제 사용처 정리** 같은 **선택적 cleanup**. 단순 이동 커밋과 섞지 않는다. `_plotly_line` 은 이번 PR 에서 `snapshot/_common.py` 로 그대로 이동만 하고, 삭제/리팩터링은 후속 cleanup PR 에서 결정.
- `from __future__ import annotations` 등 스타일 일괄 정리.
- A-3 렌더러 내부 추가 분해 (예: histogram/scatter/casebook 각각 하위 함수화). 후속 PR 후보.

## 후속 작업 (별도 PR 후보)

1. **엔트리포인트 `_resolve_color_col` re-export 제거.** 트리거 조건: 저장소 전체에서 해당 심볼의 엔트리포인트 경유 사용이 0건. (a) direct import (별칭 포함), (b) attribute access (`...page_00_market_snapshot_diagnostics._resolve_color_col`), (c) cleanup PR 의 `uv run pytest -q` green 까지 모두 확인 — 상세 grep 및 safety net 논리는 `유지해야 할 불변 조건` 섹션 참고. 본 PR 머지 직후 테스트 이전이 끝나면 두 grep 모두 0건 + pytest green 이 될 것으로 예상 — 바로 다음 PR 에서 제거 가능.
2. `_render_a3` 내부를 KPI/필터/차트/케이스북 4개 하위 함수로 추가 분해.
3. `color_map` 중복 제거 및 `a3_labels` 쪽으로 이동.
4. 다른 장편 페이지(예: `page_01_overview.py`, `page_02_trade_price.py`) 에 동일한 분리 패턴 적용.

## 리스크 및 완화책

| 리스크 | 완화 |
| --- | --- |
| 순환 import (`tab_a3` ↔ `a3_filters`) | 의존 방향을 `tab_a3 → a3_filters → a3_labels` 단방향으로 고정. `a3_filters` 는 `tab_a3` 를 import 하지 않음 |
| `sys.path` 주입 누락으로 서브모듈 import 실패 | 엔트리포인트에서 기존 bootstrap 유지. 서브모듈은 절대 import 만 사용하며 path 조작 금지 |
| 위젯 key 이름 무의식적 변경 | 함수 본문을 그대로 이동. diff 에서 위젯 `key=` 인자 검색(`rg 'key="a3_'`) 으로 변경 없음을 확인 |
| 기존 테스트 경로 깨짐 | 엔트리포인트 re-export + 테스트 import 경로 갱신을 같은 PR 에 포함 |
| CSV 컬럼/인코딩 변화 | drop 컬럼 목록(`_sggCd`, `_reason_filter_key`, `_structure_filter_key`) 및 `utf-8-sig` 유지를 수동 체크리스트로 검증 |
| Rollback 비용 급증 | PR slicing: ① **커밋 1** — 파일 이동 + re-export + 테스트 경로 갱신 (순수 이관). ② **커밋 2** — `test_a3_filter_prep.py` 추가. 문제가 발견되면 PR 자체를 `git revert` 로 되돌리기 쉽게 유지. 선택적 cleanup 은 별도 PR |
| 라인수 목표 압박으로 가독성 저하 | 숫자는 soft target. 초과해도 설계 정당성이 있으면 그대로 둔다는 점을 본 plan 에 명시 |

## 롤백 전략

- 본 PR 은 **기능 추가 없는 순수 이관 + 테스트 이관 + 테스트 추가** 3개 파트로만 구성한다.
- 문제가 발생하면 **병합 전략(merge/squash/rebase) 에 관계없이** 이번 PR 이 만든 커밋(들) 또는 squash commit 한 개를 `git revert` 해 원복 가능해야 한다. 구체적 명령 예시:
  * squash merge 의 경우: `git revert <squash-commit-sha>`
  * merge commit 이 남는 경우: `git revert -m 1 <merge-commit-sha>`
  * rebase merge 의 경우: 본 PR 의 커밋 SHA 범위를 `git revert <first>^..<last>` 로 역적용
- 위의 어떤 경로든 revert 한 번에 복귀하도록, 선택적 cleanup(예: `color_map` 중복 제거, 추가 분해) 은 본 PR 에 섞지 않는다.
- re-export 는 revert 범위 최소화를 위해 엔트리포인트에 최소 라인으로 작성한다.

## 완료 기준 (DoD)

- [ ] `page_00_market_snapshot_diagnostics.py` 가 엔트리포인트 + bootstrap + `_resolve_color_col` re-export 만 남아 있다 (대략 ≤100 라인, 정확한 숫자는 soft target).
- [ ] `dashboard/pages/snapshot/` 서브패키지가 계획된 6개 파일로 존재.
- [ ] `dashboard/app.py:20` 의 동적 import 변경 없음.
- [ ] `tests/dashboard/test_a3_color_resolver.py` import 경로가 `dashboard.pages.snapshot.a3_filters` 로 갱신되고 8개 parametrize 케이스 모두 통과.
- [ ] `tests/dashboard/test_a3_filter_prep.py` 신규 파일에서 `_ordered_present_keys` 1+ 케이스, `_prepare_a3_filter_frame` 3 가지 스키마 케이스 통과. **모든 신규/이전 테스트는 hermetic — 로컬 parquet/Streamlit 런타임 의존 금지, 입력은 inline `pd.DataFrame(...)` 리터럴.**
- [ ] 엔트리포인트의 하위 호환 re-export 경유로 `_resolve_color_col` import 가능 (동일 객체성 `is` 비교 smoke 포함). `_ordered_present_keys`, `_prepare_a3_filter_frame` 은 엔트리포인트에서 `from ... import` 시 `ImportError` 로 실패 (위 negative smoke 항목).
- [ ] `dashboard/pages/snapshot/` 서브패키지 내부에 `dashboard.data_loader` import 가 없음 (`rg "dashboard\.data_loader" dashboard/pages/snapshot/` 결과 0건). 이번 refactor 의 loader boundary 는 이 서브패키지 기준이며, 저장소 전체의 repo-wide uniqueness 는 주장하지 않는다.
- [ ] `snapshot/_common.py` 가 `streamlit` 을 import 하지 않음 (`rg "^import streamlit|^from streamlit" dashboard/pages/snapshot/_common.py` 결과 0건).
- [ ] `snapshot/a3_filters.py`, `snapshot/a3_labels.py` 가 `streamlit` / `plotly` / `dashboard.data_loader` 를 import 하지 않음 (`rg -n "streamlit|plotly|dashboard\.data_loader" dashboard/pages/snapshot/a3_filters.py dashboard/pages/snapshot/a3_labels.py` 결과 0건).
- [ ] **`_region_options` 이동 정합성 가드**: (a) `page_00_market_snapshot_diagnostics.py` 에 `_region_options` 의 `def` 이 더 이상 존재하지 않음 (`rg -n "^def _region_options" dashboard/pages/page_00_market_snapshot_diagnostics.py` 결과 0건), (b) 동일 파일에 `from dashboard.pages.snapshot._common import _region_options` import 가 존재 (`rg -n "from dashboard\.pages\.snapshot\._common import.*_region_options" dashboard/pages/page_00_market_snapshot_diagnostics.py` 결과 1건), (c) `snapshot/_common.py` 에 `def _region_options` 이 정확히 1건 존재 (`rg -n "^def _region_options" dashboard/pages/snapshot/_common.py` 결과 1건).
- [ ] `uv run pytest -q` 전체 통과.
- [ ] **Pre-merge integration smoke (implementer 가 로컬 브라우저에서 수동 수행, merge blocker)**: `uv run streamlit run streamlit_app.py` 을 최소 1회 실행해 다음 항목을 확인. (agent-only 검증 대상 아님 — 실제 사람이 로컬 브라우저에서 수행)
  * Streamlit 프로세스가 에러 없이 부팅 (import / page loading 실패 없음).
  * 사이드바에서 "데이터 진단 & 시장 스냅샷" 페이지로 진입 가능 (탭 wiring 정상).
  * A-1 / A-2 / A-3 탭이 각각 최소 1회 열림 (탭 진입 시 traceback 없음).
  * A-3 탭에서 판정사유 / 단지유형 / 색상모드 위젯이 UI 에 렌더. (위젯 존재 여부만 UI 로 확인. key literal 불변성은 아래 별도 정적 blocker 로 검증.)
  * **A-3 CSV 다운로드 버튼이 노출되고 클릭 시 traceback 이 발생하지 않는지**까지만 blocker. 실제 파일 생성 여부(브라우저 다운로드 동작) 는 환경 의존적이라 blocker 에서 제외하고 post-merge 검증으로 둔다.

  > 세부 차트/수치/레이아웃 육안 검증은 **post-merge 유저 검증**으로 둔다. pre-merge 는 "부팅 + 탭 wiring + 주요 위젯 존재 + 버튼 클릭 traceback 없음" 까지의 smoke 만 blocker 로 한다.

- [ ] **Static blocker — `session_state` key literal 불변성** (agent 수행 가능, merge blocker): 아래 5개 key literal 이 refactor 이후 코드에 그대로 남아 있는지 `rg` 로 확인한다. UI 수동 smoke 로는 증명되지 않는 항목이므로 정적 검증으로 분리.
  * `rg -n 'key="snapshot_region"' dashboard/pages/page_00_market_snapshot_diagnostics.py dashboard/pages/snapshot/` → 1건
  * `rg -n 'key="snapshot_year_range"' dashboard/pages/page_00_market_snapshot_diagnostics.py dashboard/pages/snapshot/` → 1건
  * `rg -n 'key="a3_reason_filter"' dashboard/pages/snapshot/tab_a3_outliers.py` → 1건
  * `rg -n 'key="a3_structure_filter"' dashboard/pages/snapshot/tab_a3_outliers.py` → 1건
  * `rg -n 'key="a3_color_mode"' dashboard/pages/snapshot/tab_a3_outliers.py` → 1건

  key 가 소유할 파일이 이동되었으므로 각 key 의 기대 위치를 위와 같이 고정한다. 5건 모두 정확히 1건씩 매칭되어야 한다.

- [ ] **위 항목까지가 PR merge 의 blocking 기준.**
- [ ] 상세 UI 수동 체크리스트(위 "수동 (Streamlit)" 섹션)는 **PR merge 이후 post-implementation validation** 으로 유저가 직접 수행. 이상 발견 시 아래 triage 정책에 따라 대응.

### Post-merge 이상 발견 시 triage 정책

| 증상 | 대응 |
| --- | --- |
| `streamlit_app.py` import / page load 실패 | **즉시 revert** (롤백 전략 섹션) |
| A-1/A-2/A-3 탭 진입 시 traceback | **즉시 revert** |
| widget key / session_state 회귀 (기존 필터 상태 휘발, key 이름 변경 등) | **즉시 revert** |
| A-3 CSV 다운로드 버튼 미노출 / 클릭 시 traceback / CSV payload(`st.download_button` 의 `data=`) 생성 실패 (예: `df.to_csv(...)` 예외) | **즉시 revert** |
| 버튼 클릭은 되고 payload 도 생성되지만 **브라우저 측 다운로드만 실패** (예: 로컬 브라우저/다운로드 권한/환경 이슈 정황) | 환경 이슈 여부 먼저 재현 확인 → 재현되면 hotfix PR 또는 운영 가이드 업데이트, 재현 안 되면 환경 이슈로 분류하고 조치 없음 |
| `dashboard/app.py:20` 동적 import 실패 / `render_snapshot` import 불가 | **즉시 revert** |
| `uv run pytest -q` red (post-merge 재실행 시) | **즉시 revert** |
| 차트 시각 차이 (색상 팔레트, 높이, 라벨 위치 등 비기능적) | hotfix PR |
| `st.info` / `st.caption` 본문 누락·오타 | hotfix PR |
| 문서/주석 오타 | hotfix PR |
| 불필요한 엔트리포인트 import 잔존 (`pandas`/`plotly` 등 drift) | hotfix PR |

> 판단 기준은 "user-facing 동작 / 기존 테스트 계약 / session_state 불변이 깨지면 revert, 시각·문서 차이만이면 hotfix".
