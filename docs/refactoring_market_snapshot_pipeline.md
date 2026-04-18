# `pipelines/market_snapshot_pipeline.py` 리팩토링 계획

## 배경 및 목적

- 현재 [pipelines/market_snapshot_pipeline.py](../pipelines/market_snapshot_pipeline.py) 는 **약 1,207 LoC** 의 단일 파일이다.
- 이 한 파일 안에 데이터 로딩, 공통 전처리, A-1/A-2/A-3 집계, A-3의 다단계 이상치 탐지 로직, 파이프라인 오케스트레이션이 모두 섞여 있다.
- AI(및 사람) 가 섹션 A 관련 일부 로직만 수정/탐색하려 해도 **파일 전체를 Read 해야 하므로 토큰 비용이 크다**.
- 목표: **파일당 LoC 를 줄여 AI 가 필요한 부분만 읽을 수 있도록 모듈을 분리**. 기능·동작은 유지한다 (behavior-preserving refactor).

## 현재 구조 분석

| 구간 | 라인 | 성격 |
|---|---|---|
| 모듈 docstring, import, 상수 (`AREA_BUCKETS`, 이상치 파라미터 등) | 1–72 | 설정 |
| `_load_all_trade`, `_load_all_rent` | 75–102 | I/O |
| `_add_region_columns`, `_add_area_bucket`, `_add_month_column` | 105–139 | 공통 전처리 |
| A-1: `build_snapshot_monthly_trade`, `build_snapshot_monthly_rent` | 146–274 | 월별 집계 |
| A-2: `build_snapshot_area_mix` | 281–355 | 면적 믹스 |
| A-3 (legacy trend): `_centered_rolling_median`, `_compute_monthly_band_frame`, `_annotate_trend_confirmation` | 362–502 | 이상치 - 추세 confirm |
| A-3 (v2 spread): `_build_cohort_paths`, `_build_complex_spreads`, `_compute_dynamic_band` | 505–807 | 이상치 - spread 기반 |
| A-3 엔트리: `build_snapshot_outliers` | 810–1125 | 이상치 - 최종 조합 |
| 오케스트레이션: `MarketSnapshotPipeline`, `__main__` | 1132–1206 | 실행 |

A-3 관련 코드가 **약 760 LoC** 로 전체의 60% 이상을 차지하며, 이 부분이 가장 복잡하고 자주 수정될 가능성이 높다.

### 현 파일이 맡고 있는 "암묵적 계약" (리팩토링 중 반드시 유지해야 하는 것)

리뷰 피드백에서 드러난 대로, 이 파일은 단순 import 타깃이 아니라 **다음 외부 진입점·계약을 동시에 짊어지고 있다**. 분리 계획은 이 네 가지를 전부 보존해야 한다.

1. **직접 CLI 실행 진입점**
   - 파일 말미에 `if __name__ == "__main__": pipeline.run()` ([market_snapshot_pipeline.py:1204](../pipelines/market_snapshot_pipeline.py)).
   - 파일 상단에 `sys.path` bootstrap ([market_snapshot_pipeline.py:22-25](../pipelines/market_snapshot_pipeline.py)) 이 있어, `uv run python pipelines/market_snapshot_pipeline.py` 로 직접 실행 가능.
   - 대시보드 안내 문구에도 **그 명령이 그대로 노출**되어 있음 ([dashboard/pages/page_00_market_snapshot_diagnostics.py:404](../dashboard/pages/page_00_market_snapshot_diagnostics.py)).
2. **Private helper 의 외부 import**
   - 테스트가 private helper 를 직접 import: `_add_area_bucket`, `_add_region_columns`, `_build_cohort_paths`, `_build_complex_spreads`, `_compute_dynamic_band` ([tests/test_market_snapshot_pipeline.py:6-14](../tests/test_market_snapshot_pipeline.py)).
   - 즉 "`build_snapshot_outliers` 만 공개" 수준으로 좁히면 테스트가 즉시 깨진다.
3. **산출 parquet 5 종**
   - `snapshot_monthly_trade.parquet`, `snapshot_monthly_rent.parquet`, `snapshot_area_mix.parquet`, `snapshot_outliers.parquet`, **`snapshot_complex_market_price.parquet`** ([market_snapshot_pipeline.py:1184-1199](../pipelines/market_snapshot_pipeline.py)).
   - 모듈 docstring 은 3 종만 언급해 **이미 실제와 어긋난 상태** ([market_snapshot_pipeline.py:1-11](../pipelines/market_snapshot_pipeline.py)).
4. **`run()` 내부에서 같은 모듈의 함수를 직접 호출**
   - `MarketSnapshotPipeline.run()` 이 `build_snapshot_monthly_trade/rent`, `build_snapshot_area_mix`, `build_snapshot_outliers` 를 모두 같은 파일에서 부른다 ([market_snapshot_pipeline.py:1185-1197](../pipelines/market_snapshot_pipeline.py)).
   - `runner` 를 먼저 떼어내면, A-3 가 아직 legacy 파일에 남아 있는 동안 runner 가 다시 legacy 파일을 import 하게 되어 순환·2중경로 위험.

## 분리 후 목표 구조

`pipelines/market_snapshot/` 패키지로 전환한다. 기존 `pipelines/market_snapshot_pipeline.py` 경로는 **전환 기간 내내 실제 실행 진입점 + 호환 layer** 로 유지하고, 마지막 단계에서만 얇은 shim 으로 축소한다.

```
pipelines/
├── market_snapshot_pipeline.py         # 전환 중: wrapper. 최종: shim + main() + sys.path bootstrap
└── market_snapshot/
    ├── __init__.py                     # public + legacy-private API 재노출
    ├── config.py                       # 상수·파라미터 모음 (~60 LoC)
    ├── io.py                           # _load_all_trade / _load_all_rent (~40 LoC)
    ├── preprocess.py                   # region / area_bucket / month 파생 (~50 LoC)
    ├── snapshot_monthly.py             # A-1 매매·전월세 월별 집계 (~140 LoC)
    ├── snapshot_area_mix.py            # A-2 면적 믹스 (~80 LoC)
    ├── outliers/
    │   ├── __init__.py                 # build_snapshot_outliers + private helper 재노출
    │   ├── _smoothing.py               # _centered_rolling_median / _smooth_group* 공용 helper (~40 LoC)
    │   ├── trend_band.py               # legacy band + trend confirmation (~160 LoC)
    │   ├── cohort_paths.py             # C1/C2 경로 계산 (~60 LoC)
    │   ├── complex_spreads.py          # G0/G1/G2 spread + shrinkage (~180 LoC)
    │   ├── dynamic_band.py             # band_pct / band_abs_m2 (~120 LoC)
    │   └── pipeline.py                 # build_snapshot_outliers 엔트리 (~200 LoC)
    └── runner.py                       # MarketSnapshotPipeline (~70 LoC)
```

예상 최대 단일 파일 LoC: **~200** (현재 1,207 → 약 1/6).

## 분리 기준

1. **실행 순서가 아닌 책임(역할)로 분리.** I/O, 설정, 전처리, 집계, 이상치 탐지, 오케스트레이션 을 섞지 않는다.
2. **A-3 내부는 "데이터 흐름 단계" 단위로 분리.**
   - 코호트 경로 (cohort) → 단지 스프레드 (complex spread) → 동적 밴드 (dynamic band) → 최종 판정 (pipeline) → 추세 확인 (trend band, legacy).
   - 각 모듈은 입력 DataFrame 과 출력 DataFrame 만 주고받고, 전역 상태를 갖지 않는다.
3. **공용 private helper 는 leaf 로 먼저 추출.** `_centered_rolling_median`, `_smooth_group`, `_smooth_group_col` 은 A-3 여러 단계에서 쓰이므로 `outliers/_smoothing.py` 를 **A-3 분리 시작 전에 먼저** 만들어 중복 구현·교차 import 를 막는다.
4. **상수는 `config.py` 에 모은다.** 단, A-3 전용이 명확한 상수는 `outliers/_constants.py` 로 한 번 더 좁혀도 된다 (선택).
5. **AI 탐색 비용 관점**: 자주 같이 읽히는 함수는 같은 파일에, 독립적으로 수정 가능한 함수는 분리.

## 호환성 전략

호환 대상은 네 축이다. 각각 어떻게 보존하는지 명시한다.

| 외부 계약 | 현재 | 전환 중 | 최종 |
|---|---|---|---|
| `from pipelines.market_snapshot_pipeline import MarketSnapshotPipeline, build_snapshot_*` | 직접 정의 | legacy 파일이 **실 구현 + 재export 혼재** | shim 에서 **re-export** |
| `from pipelines.market_snapshot_pipeline import _add_area_bucket, _add_region_columns, _build_cohort_paths, _build_complex_spreads, _compute_dynamic_band` (테스트) | 직접 정의 | legacy 파일에서 계속 import 가능 | shim 에서 **private helper 까지 re-export** (테스트 이전 전까지 유지) |
| `uv run python pipelines/market_snapshot_pipeline.py` CLI | 파일 하단 `__main__` + `sys.path` bootstrap | 그대로 유지 | shim 에도 **`sys.path` bootstrap + `if __name__ == "__main__": MarketSnapshotPipeline().run()`** 남김 |
| 대시보드 안내 문구의 CLI 경로 | 현 경로 그대로 | 변경 없음 | 문구 유지(경로 그대로) 또는 동일 PR 에서 동시 수정 |

최종 shim 의 골격 (최종 PR 에서만 이 형태로 축소):

```python
# pipelines/market_snapshot_pipeline.py  (최종)
from __future__ import annotations
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipelines.market_snapshot.runner import MarketSnapshotPipeline
from pipelines.market_snapshot.snapshot_monthly import (
    build_snapshot_monthly_trade,
    build_snapshot_monthly_rent,
)
from pipelines.market_snapshot.snapshot_area_mix import build_snapshot_area_mix
from pipelines.market_snapshot.outliers import build_snapshot_outliers
# 테스트 호환: private helper 재노출
from pipelines.market_snapshot.preprocess import _add_area_bucket, _add_region_columns
from pipelines.market_snapshot.outliers.cohort_paths import _build_cohort_paths
from pipelines.market_snapshot.outliers.complex_spreads import _build_complex_spreads
from pipelines.market_snapshot.outliers.dynamic_band import _compute_dynamic_band

__all__ = [
    "MarketSnapshotPipeline",
    "build_snapshot_monthly_trade",
    "build_snapshot_monthly_rent",
    "build_snapshot_area_mix",
    "build_snapshot_outliers",
]

if __name__ == "__main__":
    MarketSnapshotPipeline().run()
```

> 테스트 파일을 새 경로로 이전할지 여부는 별도 결정. 이전을 **같은 PR 에서 같이** 할 경우 shim 의 private 재export 블록은 제거 가능.

## 작업 단계 (3 PR 로 분할)

리뷰 피드백의 핵심: **runner 를 먼저 옮기지 말고, 기존 파일을 wrapper 로 남긴 채 leaf 부터 빼낸다.**

### PR 1 — 비-A3 leaf 추출 (저위험)

- 신설: `market_snapshot/{config,io,preprocess,snapshot_monthly,snapshot_area_mix}.py`.
- 기존 `pipelines/market_snapshot_pipeline.py` 는 **그대로 두고** 해당 함수 본문만 신규 모듈로 이동 → legacy 파일에서는 `from pipelines.market_snapshot.xxx import ...` 로 re-import.
- **상수 단일 source-of-truth**: `AREA_BUCKETS`, `PREPROCESSED_PLUS_DIR`, `OUTLIER_THRESHOLD`, `LOOKBACK_MONTHS`, `BOLLINGER_*`, `TREND_*`, `PATH_*`, `SPREAD_*`, `SHRINK_K`, `BAND_Z`, `FLOOR_PCT_*`, `SANITY_LOG_RATIO`, `LEADER_*`, `OLD_COMPLEX_AGE`, `RENOVATION_*` 등 ([market_snapshot_pipeline.py:29-72](../pipelines/market_snapshot_pipeline.py)) 전부를 `market_snapshot/config.py` 로 **이동(복제 금지)**. legacy 파일은 같은 PR 에서 해당 상수 로컬 정의를 **제거하고** `from pipelines.market_snapshot.config import ...` 로 가져오도록 수정. 이중 정의가 단 한 시점도 공존하지 않게 한다.
- `MarketSnapshotPipeline` 및 `build_snapshot_outliers` (+ 그 내부 private helper) 는 **이 PR 에서 움직이지 않음** → 순환·2중경로 없음.
- `__main__`, `sys.path` bootstrap **유지**.
- 회귀 테스트 (아래 "검증 체크리스트") 통과 후 머지.

### PR 2 — A-3 분리

- 순서 고정:
  1. `outliers/_smoothing.py` 에 `_centered_rolling_median`, `_smooth_group`, `_smooth_group_col` 공용 helper 먼저 추출.
  2. `outliers/cohort_paths.py` → `complex_spreads.py` → `dynamic_band.py` → `trend_band.py` → `pipeline.py` 순으로 이동.
  3. `outliers/__init__.py` 는 `build_snapshot_outliers` **와 테스트가 의존하는 private helper (`_build_cohort_paths`, `_build_complex_spreads`, `_compute_dynamic_band`) 를 함께** 노출.
- legacy 파일은 여전히 wrapper. A-3 함수들은 `from pipelines.market_snapshot.outliers import ...` 로 재노출만 함.
- `__main__` 유지.

### PR 3 — runner 이전 & shim 축소

- `market_snapshot/runner.py` 로 `MarketSnapshotPipeline` 이동.
- 기존 `pipelines/market_snapshot_pipeline.py` 를 위 "최종 shim 골격" 으로 축소.
- **같은 PR 에서 함께**:
  - 모듈 docstring 을 parquet 5 종 기준으로 현행화 (현재 3 종만 언급되어 실제와 불일치).
  - `docs/architecture.md` 업데이트.
  - `dashboard/pages/page_00_market_snapshot_diagnostics.py:404` 안내 문구가 여전히 동작하는지 smoke 확인 (CLI 경로 자체는 유지되므로 문구 변경은 선택).
  - 원하면 테스트의 import 경로를 신규 패키지 경로로 이전 — 이 경우 shim 의 private 재export 제거 가능.

## 검증 체크리스트

### Baseline 스냅샷 (리팩토링 착수 전 1회)

1. 실환경에서 `uv run python pipelines/market_snapshot_pipeline.py` (또는 `scripts/run_full_pipeline.py`) 실행.
2. 아래 **5 종** parquet 을 기준으로 둔다:
   - `snapshot_monthly_trade.parquet`
   - `snapshot_monthly_rent.parquet` *(입력 rent parquet 부재 시에만 조건부 스킵)*
   - `snapshot_area_mix.parquet`
   - `snapshot_outliers.parquet`
   - `snapshot_complex_market_price.parquet`
3. 각 parquet 에 대해 다음을 json 으로 기록:
   - `shape`, 컬럼 리스트 **(순서 포함)**, 컬럼별 dtype, 컬럼별 null count.
   - 수치 컬럼별 `describe()` (mean/std/min/25/50/75/max).
   - 범주 컬럼별 `value_counts(dropna=False)` 상위 20.
   - A-3 의 경우 추가로: `outlier_reason.value_counts()`, `reference_type.value_counts()`, `is_outlier=True` 행수, `renovation_buffer_applied=True` 행수.
   - **정렬 고정 후 전체 frame hash**: 모든 컬럼을 키로 `sort_values` → `pd.util.hash_pandas_object(df, index=False).sum()` (또는 parquet bytes 해시). 상위 N 행 해시만으로는 부족하므로 사용하지 않음.

### 각 PR 머지 기준

- 위 스냅샷을 재생성했을 때 **5 종 모두 동치**:
  - shape, 컬럼 순서, dtype, null count 완전 일치.
  - `describe()`, `value_counts()`, 카테고리별 카운트 완전 일치.
  - 정렬 후 frame hash 일치. 불일치 시 `pd.testing.assert_frame_equal(left.sort_values(...).reset_index(drop=True), right.sort_values(...).reset_index(drop=True), check_like=False)` 로 첫 불일치 지점 조사.
- `uv run pytest` 전부 통과 — 특히 `tests/test_market_snapshot_pipeline.py` 가 private helper 를 import 하므로, 이 테스트가 **경로 수정 없이** 통과해야 함.
- `rg "from pipelines.market_snapshot_pipeline"` 로 찾은 모든 호출부가 수정 없이 동작.
- CLI 실행 경로 smoke: `uv run python pipelines/market_snapshot_pipeline.py` 가 그대로 동작.
- **Consumer smoke**: 산출 parquet 는 대시보드가 바로 읽어 `month` 를 `pd.to_datetime` 으로 강제 변환한다 ([dashboard/data_loader.py:176-183](../dashboard/data_loader.py)). 각 PR 머지 직전, 아래 loader 들을 실제로 호출해 **비어있지 않은 DataFrame 이 반환되고 `month` dtype 이 `datetime64[ns]` 인지** 확인한다:
  - `load_snapshot_monthly_trade()`
  - `load_snapshot_monthly_rent()` (입력 데이터가 있을 때)
  - `load_snapshot_area_mix()`
  - `load_snapshot_outliers()`
  - `load_snapshot_complex_market_price()`

  이것으로 "집계 산출 → 대시보드 소비" 경로까지 닫는다.
- 새 파일 중 **200 LoC 초과 없음** 확인.

## 리스크 & 주의

- **동작 변경 금지**: helper 이동 시 상수 기본값, numpy dtype, `groupby(..., observed=True)`, merge 시 `how`/`on` 순서, `sort_values` 키 순서까지 그대로 유지. 이 파일은 특히 `observed=True` groupby 와 merge 후 `sort_values` 결과에 민감해서, 정렬 키 누락 하나로 회귀 hash 가 깨질 수 있다.
- **진입점 보존**: PR 1·2 에서는 `pipelines/market_snapshot_pipeline.py` 가 여전히 실행 가능해야 하며, PR 3 의 축소 shim 도 `sys.path` bootstrap 과 `__main__` 을 가져야 한다.
- **순환 import 방지**: 의존 방향은 `runner → outliers → {cohort_paths, complex_spreads, dynamic_band, trend_band} → _smoothing, preprocess → config` 단방향. `outliers/*` 가 `runner` 나 `snapshot_monthly` 를 절대 import 하지 않도록 PR 리뷰 체크리스트에 명시.
- **내부 import 규칙 (강제)**: 패키지 내부 코드는 **sibling concrete module 만 직접 import** 한다 (예: `from pipelines.market_snapshot.outliers.cohort_paths import _build_cohort_paths`). `pipelines.market_snapshot`, `pipelines.market_snapshot.outliers` 의 `__init__.py`, 그리고 legacy shim `pipelines.market_snapshot_pipeline` 은 **외부 호환용 re-export 전용**으로, 패키지 내부에서는 절대 거쳐 import 하지 않는다. 내부에서 이 경로를 타면 `run()` → outliers 호출 그래프 ([market_snapshot_pipeline.py:1156,1197](../pipelines/market_snapshot_pipeline.py)) 와 결합해 순환 import 가 재발할 수 있다. `rg "from pipelines.market_snapshot_pipeline|from pipelines.market_snapshot import" pipelines/market_snapshot` 결과가 빈 목록이어야 PR 통과.
- **loguru 로그 메시지**: 집계 통계 로그는 grep 대상일 수 있으므로 문자열 그대로 유지.
- **parquet 스키마**: `keep_cols` 목록과 컬럼 순서가 대시보드에서 사용될 수 있으므로 변경하지 않는다.
- **테스트의 private import**: PR 2 머지 후 shim 이 `_build_cohort_paths` 등을 재노출하는 동안에만 테스트가 통과한다. PR 3 에서 shim 을 축소할 때 반드시 해당 재export 블록을 남기거나, 같은 PR 에서 테스트 import 경로를 변경해야 한다. 둘 중 어느 쪽인지 PR 3 착수 시 **명시적으로 선택**한다.

## 문서·안내 문구 업데이트 범위 (PR 3 체크리스트)

- [ ] [pipelines/market_snapshot_pipeline.py](../pipelines/market_snapshot_pipeline.py) 상단 docstring: 산출물 3 종 → **5 종** 으로 현행화.
- [ ] [docs/architecture.md](architecture.md): 새 패키지 구조 반영.
- [ ] [dashboard/pages/page_00_market_snapshot_diagnostics.py:404](../dashboard/pages/page_00_market_snapshot_diagnostics.py) 안내 코드 블록 동작 확인 (경로 유지).
- [ ] `scripts/run_full_pipeline.py`, `scripts/build_summary.py` 에서 legacy import 경로가 여전히 유효한지 확인.

## 기대 효과

- AI 가 "A-1 만 고치기" / "이상치 밴드 폭만 튜닝" 같은 국소 작업 시 읽어야 할 파일이 **1,207 LoC → 60~200 LoC** 로 축소.
- 모듈 경계가 명확해져 향후 A-4 이후 지표 추가 시 충돌 없이 확장 가능.
- 동작·공개 API·CLI 진입점·테스트 private import 는 동일하므로 대시보드·스크립트·테스트 수정 없이 머지 가능.
