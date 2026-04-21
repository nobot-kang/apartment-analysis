# A-3 이상치 분류 기준 업데이트 계획 (리뷰 반영판)

## 배경

현재 [`pipelines/market_snapshot/outliers/pipeline.py`](pipelines/market_snapshot/outliers/pipeline.py) 의 `build_snapshot_outliers()` 는 1층 거래를 탐지 대상에서 제외하고, 노후 단지(築 20년+) 고가 거래는 `renovation_buffer` 로 해제한다. 이번 변경에서는 두 가지 **예외 조항**을 추가하고, **절대 금액 기준**의 새로운 판정 조건을 더한다.

### 변경 요약

1. **예외 조항 — force-false (Stage 4 뒤, 최종 플래그 확정 직전)**
   - 집계 데이터의 첫 월(= `START_YM`, 기본 2010-01) 거래는 이상치로 분류하지 않는다.
   - `age == 0` 이고 `construction_year > 0` 인 거래(신축 첫 해 + 건축년도 유효)는 이상치로 분류하지 않는다.
2. **판정 기준 추가 — 절대 금액 이탈**
   - 기준 시세 대비 **절대차(총액) 3억 원 이상** 벗어나면 이상치로 추가 판정.

## 단위 규약 — 먼저 확정 [P1]

`price` / `price_per_m2` / `ref_price_m2` 는 **만원** 단위다 ([`pipelines/data_preprocessing.py:244`](pipelines/data_preprocessing.py:244), [`pipelines/data_preprocessing.py:225`](pipelines/data_preprocessing.py:225)). 따라서:

- `ref_total = ref_m2 * area` → 만원 단위
- `price - ref_total` → 만원 단위

**신규 임계값은 만원 단위로 표기한다**:

```python
ABS_DEVIATION_MANWON: int = 30_000   # 3억 원 = 30,000 만원
```

이름을 `ABS_DEVIATION_KRW_*` 가 아니라 `ABS_DEVIATION_MANWON` 으로 명시해 향후 단위 혼동을 막는다.

### 기존 `RENOVATION_ABS_BUFFER_KRW = 50_000_000` 의 단위 버그 수정 (본 PR 스코프)

현재 [`pipelines/market_snapshot/outliers/pipeline.py:254`](pipelines/market_snapshot/outliers/pipeline.py:254) 의 비교는 `(price - ref_total) <= 50_000_000` 인데, 좌변은 만원 단위다. `50_000_000` 만원 = 5000억 원이므로 이 조건은 사실상 항상 참 — **기존 renovation buffer 의 절대금액 가드는 무력화되어 있다** (상대 가드 `RENOVATION_REL_CAP = 0.12` 만 기능 중).

본 계획은 이 버그를 함께 수정한다. 이유:
- 본 PR 이 "3억 이상 이탈은 renovation buffer 를 초과하므로 완화되지 않는다" 를 의도된 동작으로 선언하는데, 가드가 무력화된 상태로 두면 그 전제가 깨진다.
- 단위 혼동을 같은 레이어에서 일괄 정리하는 편이 이후 회귀 예방에 유리.

#### 수정 스펙

```python
# config.py  — 이름에 단위를 박아 재혼동 차단
RENOVATION_ABS_BUFFER_MANWON: int = 5_000   # 5천만 원 = 5,000 만원
# 기존 RENOVATION_ABS_BUFFER_KRW 상수는 **삭제** (별칭으로 남기지 않는다 — 오용 방지)
```

```python
# pipeline.py — Stage 4 renovation buffer
reno_mask = (
    is_outlier_final
    & (ppm > ref_m2)
    & (age_arr.to_numpy() >= OLD_COMPLEX_AGE)
    & ((price_arr - ref_total) <= RENOVATION_ABS_BUFFER_MANWON)   # 만원 단위
    & (dev_pct_arr <= RENOVATION_REL_CAP * 100)
)
```

#### 예상 영향

- **renovation_buffer_applied 건수는 감소한다**: 기존엔 상대 가드(`≤12%`) 만 실질 필터였는데, 이제 절대 가드(`≤5천만 원`) 가 복원되므로 12% 이내지만 금액 차이가 5천만 원을 초과하는 노후 고가 거래는 더 이상 완화되지 않고 이상치로 남는다.
- 대시보드 KPI `노후단지 완화 적용` 숫자가 내려가고, 전체 이상치 건수는 소폭 증가하는 것이 정상.
- 파이프라인 로그 `renovation_buffer` 카운트의 before/after 를 기록해 변화량을 트래킹한다.

## 현행 구조 요약

```
build_snapshot_outliers(trade_df)
  ├─ [drop] floor == 1 제거
  ├─ 코호트 경로 / spread / dynamic band
  ├─ Stage 1: sanity error          (log-ratio > ln 2)
  ├─ Stage 2: band candidate        (|price - ref| / ref > band_pct)
  ├─ Stage 3: support confirmation  (back 2M + forward 6M)
  ├─ legacy trend-month robust band
  ├─ Stage 4: renovation buffer release (age ≥ 20 고가 완화)
  └─ outliers_df = evaluated[is_outlier]  ← 최종 parquet 저장 경로
```

마지막 줄이 핵심: **`is_outlier=True` 인 행만 `snapshot_outliers.parquet` 에 남는다** ([`pipelines/market_snapshot/outliers/pipeline.py:296`](pipelines/market_snapshot/outliers/pipeline.py:296)). 대시보드의 [`_prepare_a3_filter_frame`](dashboard/pages/page_00_market_snapshot_diagnostics.py:112) 도 `outlier_reason` 만 읽고 `exempt_reason` 은 보지 않는다. → **force-false 로 처리된 면제 행은 대시보드에 나타나지 않는다.**

## 변경안

### 0. 설계 결정 — 면제 행 출력 경로 [P1]

원 사용자 요구는 "이상치로 분류하지 않는다" 이다. 진단용 `exempt_reason` 표시는 원 요구가 아니므로 **계획에서 제거**한다. 결과:

- 면제 행은 `snapshot_outliers.parquet` 에 **나타나지 않는다** (기존 `is_outlier=True` 필터가 유지됨).
- 면제 효과는 `market_price_df` 의 `trade_count` 에만 드러난다 (시세 집계에는 포함).
- 사유 추적이 필요해지면 별도 parquet (`snapshot_outlier_exempt.parquet`) 추가를 후속 이슈로 검토.

이 결정에 따라 `exempt_reason` 컬럼 / 대시보드 라벨 / 케이스북 컬럼 관련 스펙을 전부 삭제한다.

### 1. 예외 조항 추가 (force-false)

**위치**: Stage 4 (renovation buffer) 직후, 최종 플래그 확정 전.

**정책**: **drop 이 아니라 force-false** — 면제 행은 시세 계산에 유효한 신호이므로 `market_price_df` 집계에 포함되어야 한다. 시세 기여는 유지, 이상치 파일에서는 제외.

#### 1-1. `age == 0` 판정의 NaN 함정 [P1]

전처리에서 `construction_year == 0` (건축년도 미상) 인 행은 `age = NaN` 으로 두지만 ([`pipelines/data_preprocessing.py:201~203`](pipelines/data_preprocessing.py:201)), 현재 Stage 4 는 `age_arr = ... fillna(0)` 로 그 NaN 을 0 으로 바꿔 사용한다 ([`pipelines/market_snapshot/outliers/pipeline.py:245`](pipelines/market_snapshot/outliers/pipeline.py:245)). **이 배열을 그대로 `== 0` 검사에 쓰면 "연식 미상" 행까지 면제되는 버그가 발생한다.**

해결: 면제 판정은 **fillna 전의 raw age** 를 기준으로 하거나, `construction_year > 0` 조건을 AND 로 걸어 방어한다. 본 계획은 명시성을 위해 **둘 다 반영**한다.

**[P1] `age` 컬럼 부재 시 내구성 유지**: 현재 구현은 `evaluated.get("age", pd.Series(0, index=evaluated.index))` 로 Series 기본값을 명시해 missing 컬럼을 허용한다. 같은 패턴을 면제 판정에도 적용해야 한다. `evaluated.get("age")` 만 쓰면 컬럼이 없을 때 `None` → `pd.to_numeric(None, errors="coerce")` 가 스칼라 NaN 을 반환하고 뒤의 `.notna()` / `&` 연산이 깨진다.

```python
raw_age = pd.to_numeric(
    evaluated.get("age", pd.Series(np.nan, index=evaluated.index)),   # 컬럼 부재 시 NaN Series
    errors="coerce",
)
construction_year = pd.to_numeric(
    evaluated.get("construction_year", pd.Series(0, index=evaluated.index)),
    errors="coerce",
).fillna(0)

age_zero_exempt_mask = (
    raw_age.notna()                # NaN (연식 미상) 은 제외
    & (raw_age == 0)
    & (construction_year > 0)      # 이중 방어
).to_numpy()
```

`age` 컬럼이 없는 입력에서는 `raw_age` 가 전부 NaN 이 되어 `age_zero_exempt_mask` 가 전부 False — 면제 대상이 없는 것으로 자연스럽게 동작한다. 기존 `age_arr` (Stage 4 renovation 용 `fillna(0)` 배열) 은 **재사용하지 않는다**.

#### 1-2. 첫 월 판정은 `START_YM` 에서 **함수 안에서** 파생 [OQ]

상수 중복 정의 방지 + monkeypatch 가 같은 프로세스에서 즉시 반영되도록, config 모듈 상수로 캐시하지 않고 **`build_snapshot_outliers()` 안에서 매 호출마다 파생**한다.

```python
# pipelines/market_snapshot/outliers/pipeline.py  (함수 내부)
from config.settings import START_YM  # 함수 선두에서 import

first_snapshot_period = pd.Period(f"{START_YM[:4]}-{START_YM[4:]}", freq="M")
```

(테스트에서 `monkeypatch.setattr("config.settings.START_YM", "201501")` 가 바로 파이프라인 실행에 반영되어야 하므로 모듈 top-level 에서 import 하지 않는다.)

#### 1-3. `renovation_buffer_applied` 플래그 의미 재정의 [P2]

현재 흐름: Stage 4 에서 renovation 완화 조건을 만족하면 `renovation_buffer_applied=True` + `is_outlier=False` 가 함께 세팅된다 ([`pipelines/market_snapshot/outliers/pipeline.py:242`](pipelines/market_snapshot/outliers/pipeline.py:242), [`:257~258`](pipelines/market_snapshot/outliers/pipeline.py:257)). 그 뒤 면제 단계가 추가되면 아래 3가지 행이 생긴다:

| 케이스 | Stage 3 `is_outlier_final` | renovation 조건 | 면제 조건 | 최종 `is_outlier` | 현재 플래그 동작 |
|---|---|---|---|---|---|
| A | True | 만족 | 불만족 | False | `renovation_buffer_applied=True` — 맞음 |
| B | True | 불만족 | 만족 | False | `renovation_buffer_applied=False` — 맞음 |
| **C** | **True** | **만족** | **만족** | **False** | **`renovation_buffer_applied=True` 로 남아 renovation 공로로 계수** |

대시보드 KPI 문구는 "renovation_buffer 로 outlier 해제된 거래 수" ([`dashboard/pages/page_00_market_snapshot_diagnostics.py:461`](dashboard/pages/page_00_market_snapshot_diagnostics.py:461)) — 케이스 C 는 renovation 이 먼저 해제했지만 **면제만으로도 해제될 행**이므로 "renovation 공로" 라고 보기 어렵다. 의미를 "면제와 무관하게 renovation 덕분에 해제된 행" 으로 좁혀 KPI 정확도를 높인다.

**결정**: 면제 조건을 만족하는 행에서는 `renovation_buffer_applied=False` 로 되돌린다 (면제가 더 근본적 해제 사유).

```python
# 면제 force-false 블록 내부
evaluated.loc[exempt_condition_mask, "renovation_buffer_applied"] = False
```

- `exempt_condition_mask` 는 "면제 조건을 만족한 전체 행" (live is_outlier 무관). 케이스 C 는 이미 `is_outlier=False` 이므로 `exempt_flipped_mask` 에는 안 잡히지만, 이 라인으로 플래그만 정리한다.
- 대시보드 KPI 는 수정 없이 그대로 유효 — 의미가 "renovation 덕분에 해제된" 으로 정확해진다.

#### 로그 스펙 잠금 [P2]

현재 [`pipelines/market_snapshot/outliers/pipeline.py:328~339`](pipelines/market_snapshot/outliers/pipeline.py:328) 의 `logger.info` 는 `n_reno = int(np.sum(reno_mask))` 로 **raw reno_mask** 를 집계한다. 플래그 의미가 재정의된 뒤에도 이 라인을 그대로 두면 **로그의 `renovation_buffer` 카운트와 대시보드의 `renovation_buffer_count` (= `renovation_buffer_applied.sum()` 기반) 가 어긋난다** — 로그는 케이스 C 를 포함하고 KPI 는 제외.

**결정 — 두 수치를 모두 로그**:

```python
# 로그 집계부
n_reno_raw       = int(np.sum(reno_mask))                                # 원시 renovation 트리거
n_reno_effective = int(evaluated["renovation_buffer_applied"].sum())     # 면제 정리 후 최종 공로 (= KPI 와 일치)

# [P1] 중복 집계 방지: 2010-01 + age==0 동시 만족 행이 두 번 세지지 않도록 mask 합집합의 sum 사용
n_exempt_condition = int(exempt_condition_mask.sum())                    # OR mask 의 sum (합집합)

logger.info(
    f"... renovation_buffer_raw={n_reno_raw}, "
    f"renovation_buffer_effective={n_reno_effective}, "
    f"first_month_exempt={n_first_month_exempt}, "       # 개별 조건 원시 카운트
    f"age_zero_exempt={n_age_zero_exempt}, "             # 개별 조건 원시 카운트
    f"exempt_condition_total={n_exempt_condition}, "     # 합집합 (중복 제거)
    f"exempt_flipped={n_exempt_flipped}, ..."
)
```

원시 개별 카운트(`first_month_exempt`, `age_zero_exempt`)도 남겨 두면 "어느 조건이 얼마나 발동했나" 를 볼 수 있고, 합집합(`exempt_condition_total`) 으로 "전체 면제 후보 수" 를 정확히 집계한다. 로그 변경 위치: [`pipelines/market_snapshot/outliers/pipeline.py:334`](pipelines/market_snapshot/outliers/pipeline.py:334).

- `raw` 와 `effective` 를 둘 다 남기면 "renovation 가드가 몇 건 트리거되었나 vs 최종 KPI" 를 같이 볼 수 있어 디버깅 정보가 보존된다.
- 대시보드 KPI 는 `renovation_buffer_effective` 와 정확히 일치 (`market_price_df.renovation_buffer_count` 는 `renovation_buffer_applied` 를 집계).

#### 1-4. 최종 스케치 — live 상태 기준 [P1]

**중요**: `is_outlier_final` (numpy 배열) 은 Stage 3 끝난 시점의 스냅샷이며 Stage 4 renovation 완화를 **반영하지 않는다**. renovation 으로 이미 `evaluated["is_outlier"] = False` 가 된 행을 "면제로 뒤집힘" 에 잘못 계수하지 않도록, 면제 적용과 `exempt_flipped` 카운트는 **Stage 4 직후의 `evaluated["is_outlier"]` (live 상태)** 를 기준으로 한다.

```python
# Stage 4 (renovation buffer) 직후
# 이 시점의 evaluated["is_outlier"] 은 renovation 완화가 이미 반영된 live 상태
live_is_outlier = evaluated["is_outlier"].to_numpy()

first_month_mask = (
    evaluated["month"].dt.to_period("M") == first_snapshot_period
).to_numpy()

exempt_condition_mask = age_zero_exempt_mask | first_month_mask   # 조건 만족 전체
exempt_flipped_mask   = live_is_outlier & exempt_condition_mask    # 이상치에서 뒤집힘

evaluated.loc[exempt_flipped_mask, "is_outlier"] = False

# renovation 공로 재정의 (1-3 참조): 면제 조건 만족 행은 renovation 공로에서 제외
evaluated.loc[exempt_condition_mask, "renovation_buffer_applied"] = False

# 카운트 (면제 조건 만족 전체 행 수 vs 실제 뒤집힌 수)
n_first_month_exempt = int(first_month_mask.sum())
n_age_zero_exempt   = int(age_zero_exempt_mask.sum())
n_exempt_flipped    = int(exempt_flipped_mask.sum())
```

면제된 행이 reason/direction 컬럼에 잔존값을 가져도 `outliers_df = evaluated[is_outlier]` 필터링에 의해 자동 제거되므로 추가 정리는 불필요.

### 2. 절대 금액 기준 이상치 추가 (±3억 원)

**위치**: Stage 1/2 와 같은 레이어.

**단위**: 만원 (위 "단위 규약" 참조).

```python
# config.py
ABS_DEVIATION_MANWON: int = 30_000   # 3억 원
```

```python
# pipeline.py — Stage 2 이후, unsupported 판정과 나란히
ref_total_m2_area = ref_m2 * evaluated["area"].to_numpy(dtype=float)  # 만원 단위
price_arr = evaluated["price"].to_numpy(dtype=float)                  # 만원 단위
abs_dev_manwon = np.abs(price_arr - ref_total_m2_area)

abs_mask = (
    np.isfinite(ref_total_m2_area) & (ref_total_m2_area > 0)
    & np.isfinite(abs_dev_manwon)
    & (abs_dev_manwon >= ABS_DEVIATION_MANWON)
)

is_outlier_v2 = sanity_mask | unsupported_mask | abs_mask
```

**`outlier_reason` 우선순위**:

```python
outlier_reason_v2 = np.where(
    sanity_mask, "sanity_error",
    np.where(unsupported_mask, "unsupported_jump",
    np.where(abs_mask, "abs_deviation", "")),
)
```

- `reference_type` 은 기존 `moving_average_band` 분기에 편입.
- 대시보드 `A3_REASON_LABELS` / `A3_REASON_ORDER` / `color_map` 에 `"abs_deviation": "절대 금액 이탈 (±3억)"` 추가 (색상은 `teal` 권장).

### 3. 케이스북 `편차(만원)` 단위 버그 동반 수정 [P2]

[`pipelines/market_snapshot/outliers/pipeline.py:290~291`](pipelines/market_snapshot/outliers/pipeline.py:290) 에서 `deviation_total_krw = price_arr - ref_price_total` 로 계산되는데, `price`/`ref_price_total` 이 이미 **만원** 단위이므로 `deviation_total_krw` 도 만원 단위다 (컬럼명은 오해의 소지가 있음).

그런데 대시보드는 [`dashboard/pages/page_00_market_snapshot_diagnostics.py:662`](dashboard/pages/page_00_market_snapshot_diagnostics.py:662) 에서 이를 다시 `1e4` 로 나눠 "편차(만원)" 로 표시한다 — 실제 값은 **억 단위**가 "만원" 으로 찍히고 있다. `abs_deviation` 판정사유가 새로 필터에 노출되면 사용자가 이 컬럼을 직접 참조할 가능성이 높아 동반 수정이 필수.

#### 수정 스펙

```python
# dashboard/pages/page_00_market_snapshot_diagnostics.py:660~663  현재
if "deviation_total_krw" in top_cases.columns:
    top_cases["deviation_total_krw"] = (top_cases["deviation_total_krw"] / 1e4).round(0).astype("Int64")
    top_cases = top_cases.rename(columns={"deviation_total_krw": "편차(만원)"})
```

→

```python
if "deviation_total_krw" in top_cases.columns:
    # 파이프라인 출력이 이미 만원 단위. 나누지 않고 표시.
    top_cases["deviation_total_krw"] = top_cases["deviation_total_krw"].round(0).astype("Int64")
    top_cases = top_cases.rename(columns={"deviation_total_krw": "편차(만원)"})
```

파이프라인 쪽 컬럼명(`deviation_total_krw`) 을 바꾸면 소비자가 더 있는지 확인 필요 (`ref_price_total` 등과 세트)이므로 본 PR 에서는 **대시보드 표시 로직만** 수정한다. 컬럼명 리네이밍은 후속 이슈.

### 4. 예외 조항과 신규 판정의 상호작용

- 면제(first_month / age==0)는 **모든 `is_outlier_*` 마스크 OR 결합 결과**를 덮어쓰므로 `abs_mask` 도 면제 대상. 의도된 동작.
- `renovation_buffer` 는 단위 버그 수정 후 `≤ 5천만 원` + `≤12%` 이중 가드로 동작한다. **3억(= 30,000 만원) 이상 이탈은 5천만 원 가드를 명확히 초과하므로 `abs_mask` 로 잡힌 거래는 renovation 완화 대상이 되지 않는다** — 의도된 동작이 실제로 성립.

## 면제 카운트 정의 [OQ]

- `first_month_exempt` / `age_zero_exempt`: **조건 만족 전체 행 수** (원 outlier 여부 무관).
- `exempt_flipped`: **Stage 4 renovation 이후 live 상태에서 실제로 이상치에서 제외된 행 수**. `is_outlier_final` (renovation 이전 배열) 이 아니라 `evaluated["is_outlier"].to_numpy()` 를 사용해야 renovation 으로 이미 해제된 행이 중복 계수되지 않는다.

세 카운트 모두 `logger.info` 에 기록. 1-3 섹션 스케치 참조.

## 파일·상수 체크리스트

| 파일 | 변경 |
|---|---|
| [`pipelines/market_snapshot/config.py`](pipelines/market_snapshot/config.py) | `ABS_DEVIATION_MANWON = 30_000` 추가; `RENOVATION_ABS_BUFFER_KRW` 삭제 후 `RENOVATION_ABS_BUFFER_MANWON = 5_000` 로 교체 (상수에 `FIRST_SNAPSHOT_PERIOD` 는 두지 않음 — 함수 내부에서 `START_YM` 파생) |
| [`pipelines/market_snapshot/outliers/pipeline.py`](pipelines/market_snapshot/outliers/pipeline.py) | `START_YM` 함수 내 import → `first_snapshot_period` 파생; Stage 2 옆 `abs_mask` 계산; Stage 4 renovation 가드를 `RENOVATION_ABS_BUFFER_MANWON` 으로 교체; Stage 4 뒤 `exempt_flipped_mask` force-false (live `evaluated["is_outlier"]` 기준); `outlier_reason_v2` 3지 분기; docstring 업데이트; 카운트 로그 확장 |
| [`pipelines/market_snapshot_pipeline.py`](pipelines/market_snapshot_pipeline.py) | **[P1]** line 68 의 `RENOVATION_ABS_BUFFER_KRW` import 를 `RENOVATION_ABS_BUFFER_MANWON` 으로 교체. 누락 시 모듈 import 자체가 깨져 테스트 진입도 불가. |
| [`dashboard/pages/page_00_market_snapshot_diagnostics.py`](dashboard/pages/page_00_market_snapshot_diagnostics.py) | `A3_REASON_LABELS`/`A3_REASON_ORDER` 에 `abs_deviation` 추가; `color_map` 에 색 추가; A-3 subheader 캡션의 "사전 제외" 문구 갱신; **[P2] `deviation_total_krw` 표시 단위 버그 수정** (아래 별도 섹션) |

`exempt_reason` 관련 변경은 전부 삭제.

## 검증 항목

- [ ] `age == 0` + `construction_year > 0` + `|price - ref_total| >= 3억` → `is_outlier=False`, `outliers_df` 에서 제거됨
- [ ] `age == NaN` (연식 미상) + 3억 이탈 → `is_outlier=True` (면제되지 않아야 함 — 회귀 방지 핵심)
- [ ] `month == 2010-01` + `abs_mask` 충족 → `is_outlier=False`
- [ ] 2010-01 이후 + `age >= 1` + `|price - ref_total| >= 3억` (만원 단위 30,000) + band_pct 는 통과 → `is_outlier=True`, `outlier_reason="abs_deviation"`
- [ ] 동일 거래가 `sanity_error` 도 만족 → `outlier_reason="sanity_error"` 우선
- [ ] `ref_price_total` NaN/0 거래는 `abs_mask` 에서 자동 제외 (크래시 없이)
- [ ] `market_price_df` 의 `trade_count` 에 면제 거래가 **포함**되어야 함 (force-false 효과 확인)
- [ ] 대시보드 A-3 에서 `"절대 금액 이탈 (±3억)"` 판정사유가 필터/범례에 정상 노출
- [ ] `logger.info` 에 `first_month_exempt`, `age_zero_exempt`, `exempt_flipped` 3개 카운트 기록
- [ ] `START_YM` 을 미래값(예: "201501") 으로 일시 변경 후 재실행 시 면제 대상 월이 자동 추종 (drift 방지 검증, 수동)
- [ ] 노후 단지 + 상대 편차 10% + 절대 편차 8천만 원 거래 → 기존: 완화 적용(`is_outlier=False`), 수정 후: 완화 미적용(`is_outlier=True`) — renovation 단위 버그 수정 검증
- [ ] 노후 단지 + 상대 편차 10% + 절대 편차 3천만 원 거래 → 수정 후에도 완화 적용 유지 (5천만 원 가드 내)
- [ ] 파이프라인 실행 전/후 `renovation_buffer_count` 총합 비교 로그 확인 (감소 방향이어야 정상)
- [ ] [`pipelines/market_snapshot_pipeline.py`](pipelines/market_snapshot_pipeline.py) 가 정상 import 됨 (상수 rename 전파 확인). `uv run python -c "import pipelines.market_snapshot_pipeline"` 로 사전 검증.
- [ ] renovation 이 이미 해제한 행이 `exempt_flipped` 카운트에 중복 계수되지 않음 (수동 확인: `renovation_buffer_applied=True` 이면서 면제 조건도 만족하는 fixture 를 넣고 `exempt_flipped` 가 0 인지 확인)
- [ ] **케이스 C** (Stage 3 이상치 + renovation 완화 가능 + 면제 조건 동시 만족) → 최종 `is_outlier=False`, `renovation_buffer_applied=False` (renovation 공로에서 제외)
- [ ] `logger.info` 의 `renovation_buffer_effective` 와 `market_price_df.renovation_buffer_count.sum()` 이 **정확히 일치** (정의 drift 방지)
- [ ] 케이스 C 가 있을 때 `renovation_buffer_raw > renovation_buffer_effective` (raw 는 트리거 시점, effective 는 KPI 시점)
- [ ] `age` 컬럼이 없는 입력으로 파이프라인 실행 시 크래시 없이 완료 (면제 판정이 자연스럽게 전부 False)
- [ ] 대시보드 케이스북 `편차(만원)` 컬럼이 실제로 만원 단위로 표시됨 (수정 전: 1억 = `1`, 수정 후: 1억 = `10000`)

## 테스트 전략 [P2]

### 헬퍼 확장

기존 헬퍼는 월 시퀀스를 `2020-01` 부터 고정 생성하므로 ([`tests/test_market_snapshot_pipeline.py:24`](tests/test_market_snapshot_pipeline.py:24), [`tests/test_market_snapshot_pipeline.py:149`](tests/test_market_snapshot_pipeline.py:149)) **첫 월(2010-01) 케이스는 헬퍼 재사용으로 만들 수 없다**. 헬퍼에 `start_month` 파라미터를 추가 (default `"2020-01"` 유지로 하위 호환).

### `age` 주입 방식 [P1]

`build_snapshot_outliers()` 는 입력의 `age` 컬럼을 그대로 읽으며 재계산하지 않는다 ([`pipelines/market_snapshot/outliers/pipeline.py:245`](pipelines/market_snapshot/outliers/pipeline.py:245)). 따라서 **`construction_year` 만 바꿔서는 `age=0` / `age=NaN` 경로를 탈 수 없다**. 테스트 fixture 에 **`age` 컬럼을 명시 주입**하거나, 입력 행에 대해 전처리 `_add_metadata_columns` 를 거친 후 전달해야 한다.

본 계획은 간결성을 위해 **fixture 에 `age` 직접 주입** 방식을 선택. 헬퍼 시그니처에 `age` override 파라미터(또는 `age_by_month`) 를 추가한다.

```python
# 예시: fixture 확장
_build_trade_frame(
    ...,
    age=0,                         # 전체 행 age=0 주입
    # 또는 행별로 다르게 하려면
    ages_override={2010-01: np.nan, 2020-05: 0, ...},
)
```

### 경계 테스트 설계 지침 [P1]

`29_999` / `30_001` 경계 테스트는 세 가지 이유로 flaky + 비현실적이 될 수 있다:

1. **헬퍼가 `price = int(price_per_m2 * area)` 로 재생성**한다 ([`tests/test_market_snapshot_pipeline.py:35`](tests/test_market_snapshot_pipeline.py:35), [`tests/test_market_snapshot_pipeline.py:160`](tests/test_market_snapshot_pipeline.py:160)) — float 오차 + `int()` truncation 으로 경계값이 1만원 단위로 흔들린다.
2. `ref_total` 자체가 cohort/spread 경로에 따라 변동해 테스트 시점에 예측하기 어렵다.
3. `price_override` 만 단독 추가하면 전처리가 보장하는 **`price_per_m2 = price / area` 불변식**이 깨진다 ([`pipelines/data_preprocessing.py:244`](pipelines/data_preprocessing.py:244)). 파이프라인은 `price_per_m2` 로 band/sanity 를 판정하고 `price` 로 abs mask 를 판정하므로, 두 값이 어긋나면 production-unlike 한 fixture 가 되어 실제 회귀를 못 잡는다.

**해결 — 두 축 동시 고정**:

#### 축 1. `ref_total` 을 production-like 하게 사실상 고정 (2-apt baseline + B 자기 이력) [P1]

`spread_g0` / `spread_g1` / `spread_g2` 는 cohort 가 아니라 **같은 aptSeq 내부 집계** ([`pipelines/market_snapshot/outliers/complex_spreads.py:28`](pipelines/market_snapshot/outliers/complex_spreads.py:28), [`:41`](pipelines/market_snapshot/outliers/complex_spreads.py:41)). `shrinkage-blended spread` 는 `n_12m` 에 비례 가중되어 B 의 자체 거래 이력이 적으면 해당 월의 spread 가 그 거래 자체를 따라간다 ([`pipelines/market_snapshot/outliers/complex_spreads.py:115~133`](pipelines/market_snapshot/outliers/complex_spreads.py:115)). 따라서 "A 로만 cohort 를 고정하고 B 는 단일 경계 거래 1건" 설계는 `ref_price_m2` 가 B 의 경계값 쪽으로 끌려 전제가 깨진다.

**고정된 fixture 구조**:

- **앵커 단지 A** (≥1, 동일 cohort = 같은 `sggCd` × `area_bucket`):
  - 12개월 × 여러 건 평탄 거래 → cohort 경로(c1/c2) 고정.
- **테스트 단지 B** (경계 거래 대상):
  - **과거 평탄한 자기 이력** 을 최소 6~12개월, 매월 1~2건, 동일 `price_per_m2 = baseline_ppm` 으로 삽입. 이 시점 spread_g0 ≈ log(baseline_ppm) - log(path_cohort) 로 안정.
  - **마지막 월(target_month)에 1건만 경계 거래** (`price_manwon = ref_total_expected ± delta`).
  - `n_12m` 이 충분히 크면 shrinkage weight 가 커져 spread_shrunk 이 장기 평균 쪽으로 잡히고, target_month 거래 1건이 spread 에 미치는 영향이 희석된다.
- 테스트는 `ref_total_expected = baseline_ppm * area` 로 추정한 값에 delta 를 더한다. B 자기 이력을 6개월 이상 평탄하게 두면 ref 편차가 1만원 단위(경계의 1 단위) 이하로 흡수된다.

#### 축 2. 헬퍼는 `price` 와 `price_per_m2` 를 **함께** 일관되게 세팅

`price_override` 단독이 아니라 **세트 입력**으로 확장:

```python
# 헬퍼 확장 시그니처 (권장)
def _build_trade_row(
    ...,
    price_manwon: int | None = None,       # 지정 시 price_per_m2 = price / area 로 자동 파생
    # default: 기존 동작 (price_per_m2 로부터 price 재계산)
):
    if price_manwon is not None:
        price = int(price_manwon)
        price_per_m2 = round(price / area, 2)   # 전처리 불변식 유지
    else:
        price_per_m2 = ...                       # 기존
        price = int(price_per_m2 * area)         # 기존
```

이렇게 하면 호출자가 `price_manwon` 하나만 넘겨도 `price_per_m2` 가 자동으로 동기화되어 전처리 불변식이 깨지지 않는다. abs 경계 테스트는:

```python
baseline_ppm = 2_000           # 만원/㎡ — 평탄한 baseline
area = 100                     # ㎡ — 단위당 노이즈 흡수
ref_total_expected = baseline_ppm * area   # = 200_000 만원

# 경계: delta = 29_999 → 이상치 아님, delta = 30_001 → 이상치
test_price = ref_total_expected + 29_999   # 또는 + 30_001
_build_trade_row(..., area=area, price_manwon=test_price)
```

baseline 이 평탄하면 `ref_total` 과 `ref_total_expected` 의 차이는 1만원 단위 이하로 흡수되어 경계 판정이 결정적이다.

#### 추가 조건

- `area` 를 크게 (예: 100㎡) 잡아 단위당 노이즈 흡수.
- baseline `ref_price_m2` 시계열을 **완전 평탄**하게 (6개월 연속 동일값, 테스트 단지 거래 월 앞뒤 지지 거래 없음).
- target delta 를 band_pct 이내로 잡아 band 판정은 통과, abs 마스크만 단독 트리거.
- 경계 테스트에서는 delta 를 `29_999` / `30_001` 로 **정수** 로 전달 (float 경유 X).

### 추가 테스트 케이스

1. **`test_exempt_first_month`** — `start_month="2010-01"` 로 시리즈 생성, 첫 월에 극단 고가 거래 삽입 → `is_outlier=False`, `outliers_df` 에 없음. `monkeypatch.setattr("config.settings.START_YM", "201001")` 도 함께 (default 이지만 명시적으로).
2. **`test_first_month_follows_start_ym_monkeypatch`** — `monkeypatch` 로 `START_YM="201501"` 설정 후, `2015-01` 거래가 면제되고 `2010-01` 은 면제되지 않음을 확인 (함수 내부 파생 검증).
3. **`test_exempt_age_zero`** — fixture 에 **`age=0` 직접 주입** + `construction_year > 0` + 3억 이탈 거래 → 제외됨.
4. **`test_age_nan_not_exempted`** *(회귀 방지 핵심)* — fixture 에 **`age=np.nan` 직접 주입** (또는 `construction_year=0` + `age=NaN`) + 3억 이탈 → `is_outlier=True` 유지.
5. **`test_abs_deviation_triggers_outlier`** — age≥1, 2010-01 이외, `|price - ref_total| >= 30_000` (만원) 이지만 band_pct 통과 → `is_outlier=True`, `outlier_reason="abs_deviation"`. 위 "경계 테스트 설계 지침" 적용.
6. **`test_abs_deviation_priority_under_sanity`** — sanity+abs 동시 성립 → `outlier_reason="sanity_error"`.
7. **`test_market_price_includes_exempt`** — 면제 거래가 `market_price_df.trade_count` 에 계수됨.
8. **`test_abs_threshold_unit_is_manwon`** *(단위 회귀 방지)* — 경계 설계 지침(축 1 + 축 2) 적용. 2-apt baseline + `price_manwon=ref_total_expected + 29_999` → False, `+ 30_001` → True. **헬퍼는 `price` 와 `price_per_m2` 를 함께 세팅해 전처리 불변식 유지**.
9. **`test_renovation_abs_buffer_unit_is_manwon`** *(renovation 단위 회귀 방지)* — 아래 "renovation 테스트 진입 경로" 참조. age≥20 + `trend_month_robust_band` 로 `is_outlier=True` 확보 + 상대 편차 10% (REL_CAP 12% 이내) + 절대 편차 6천만 원(만원 단위 `6_000`) → `renovation_buffer_applied=False`, `is_outlier=True` (가드 복원 확인).
10. **`test_renovation_abs_buffer_within_limit`** — 같은 진입 경로 + 절대 편차 4천만 원 → 완화 적용 (`is_outlier=False`).
11. **`test_renovation_flag_cleared_for_exempt_row`** *(케이스 C 회귀 방지)* — "renovation 테스트 진입 경로" 적용 + 대상 월을 `2010-01` 로 설정 (면제 대상). age≥20 + trend_row outlier + 절대 편차 4천만 원 → `is_outlier=False` (면제), `renovation_buffer_applied=False` (공로 제거), `market_price_df.renovation_buffer_count` 에도 계수되지 않음. 로그 수치 검증은 아래 "로그 검증 방식" 참조.

### renovation 테스트 진입 경로 [P1]

9·10 케이스가 renovation 단계를 타려면 먼저 `is_outlier_final=True` 가 되어야 한다. 그러나:

- Stage 2 (`band candidate`) 는 `|price - ref| / ref > band_pct` 로 판정하고, 기본 band floor 는 `FLOOR_PCT_BASE = 0.18` ([`pipelines/market_snapshot/outliers/dynamic_band.py:99`](pipelines/market_snapshot/outliers/dynamic_band.py:99)) — 상대 편차 10% 만으로는 Stage 2 candidate 자체가 되지 않는다.
- renovation 완화의 `RENOVATION_REL_CAP = 0.12` 이내 조건 ([`pipelines/market_snapshot/outliers/pipeline.py:255`](pipelines/market_snapshot/outliers/pipeline.py:255)) 을 만족하려면 상대 편차를 12% 이내로 유지해야 함.
- 두 조건이 상충 (band floor 18% vs REL_CAP 12%) → **Stage 2 경로로는 renovation 에 들어갈 fixture 를 만들 수 없다**.

**해결 — `trend_month_robust_band` 경로로 진입** ([`pipelines/market_snapshot/outliers/pipeline.py:216~224`](pipelines/market_snapshot/outliers/pipeline.py:216)):

#### `trend_confirmed` 정확 조건 복기 [P2]

`trend_row_outlier = trend_confirmed & (month_trade_count >= 3) & |price - month_price_m2| > month_row_band_abs`. `trend_confirmed=True` 는 breakout 월과 그 support 월에만 붙는다 ([`trend_band.py:92~159`](pipelines/market_snapshot/outliers/trend_band.py:92)):

- **breakout 월**: `month_price_m2` 가 `ref_price ± band_width_abs` 밖 (`candidate_direction ≠ 0`, [`:80`](pipelines/market_snapshot/outliers/trend_band.py:80)).
- **support 월 2개 이상** (lookhead 6개월): `direction × (future_price - ref) >= band × TREND_SUPPORT_BAND_RATIO` (=0.5).
- `total_trades ≥ TREND_MIN_TOTAL_TRADES` (3) — breakout + supports 합계.
- breakout 월 가격이 새 레벨 대비 `±TREND_ALIGNMENT_TOLERANCE` (12%) 이내.

이 조건을 통과하면 breakout 월**과** support 월 전부 `trend_confirmed=True`.

**리뷰 지적의 본질**: 기존 스케치("대상 월에 기준가 근처 2건 + target 1건")는 대상 월의 `month_price_m2` = median 이 기준가 근처로 잡혀서 **candidate_direction=0** 이 되고, 이전 breakout 도 없으면 support 도 아니다 → `trend_confirmed` 가 안 붙는다.

#### 확정 fixture 레이아웃 [P2]

"**이전 월에서 breakout, 대상 월은 support**" 레이아웃으로 고정:

```
month t-6 ~ t-2  : baseline_level 에서 평탄 거래 (월 1~2건) → ref_price / band 형성
month t-1        : breakout 월 — 월 median 을 new_level (= baseline × 1.30) 로 세팅, 2건
month t (target) : support 월 — 대상 월 내부 3건 배치
                    (a) 2건은 new_level 근처 → month_price_m2 ≈ new_level
                    (b) 1건은 row outlier target: new_level × 1.10 + 절대 delta 조정
                    direction × (new_level - ref_price) >= band × 0.5 를 만족해 support 자격
month t+1 ~      : 추가 support 월 최소 1개 (new_level 근처) → support_months ≥ 2 확보
```

- **breakout 크기 [P2]**: legacy breakout 판정은 `band_width_abs = max(rolling_std*2, ref_price * OUTLIER_THRESHOLD)` 이고 (`OUTLIER_THRESHOLD = 0.25`, [`config.py:25`](pipelines/market_snapshot/config.py:25), [`trend_band.py:67`](pipelines/market_snapshot/outliers/trend_band.py:67)) 후보 조건은 **strict `>`** (`month_price_m2.gt(band_upper)`, [`trend_band.py:82`](pipelines/market_snapshot/outliers/trend_band.py:82)). baseline 이 평탄할 때 `band_upper = ref_price × 1.25` 이므로 `new_level = baseline × 1.25` 는 경계에 "딱 걸쳐" `candidate_direction=0` 이 될 수 있다 → `trend_confirmed` 미부착. **안전 마진을 위해 `new_level = baseline × 1.30` 으로 고정** (상한 `1.25x` 대비 +5%p 마진, 헬퍼 수치 오차 + `rolling_std*2` 가드 양쪽 모두 흡수).
- 대상 월의 `month_price_m2` 는 (a) 2건의 영향으로 new_level 근처. target 거래 (b) 는 `month_price_m2` 대비 10% 높음 → `month_row_band_abs` 를 초과하도록 delta 설계 (`TREND_ROW_STD_MULTIPLIER`, `TREND_ROW_MIN_BAND_PCT=0.08` 참조).
- target 거래의 **spread-based ref_price_m2 대비 상대 편차는 ≤ 12%** 로 유지 (renovation REL_CAP 안쪽). new_level 이 cohort 대비 30% 위에 있어도, target 거래는 new_level 기준으로는 10% 이탈이고 spread_shrunk 가 new_level 쪽으로 잡히면 `ref_price_m2 ≈ new_level` 이 되어 target 의 상대 편차가 10% 근처로 맞춰진다 (cohort 가 함께 추세 이동해 있다면).
- cohort 가 따라 움직이지 않으면 target 의 cohort 대비 편차가 40%+ 가 되어 Stage 2 candidate 로도 잡혀 renovation REL_CAP 12% 를 넘어 완화 대상에서 빠진다. **이를 피하려면 fixture 에 동일 cohort (동일 `sggCd × area_bucket`) 의 다른 단지들도 같은 추세 상승**을 주입하여 cohort 경로가 함께 이동하게 한다.
- 절대 편차는 target 거래의 `price_manwon` 을 `ref_total_expected` (= new_level × area) 에서 `+6_000` (만원) 또는 `+4_000` 으로 직접 주입 (축 2 `price_manwon` 헬퍼 경로).

##### 구체 수치 조합 — 테스트별 [P3]

renovation 게이트는 `ref_m2` 기준 상대 편차 ≤12% **그리고** 절대 편차 ≤ `RENOVATION_ABS_BUFFER_MANWON` (=5,000) 을 AND 로 본다 ([`pipeline.py:248`](pipelines/market_snapshot/outliers/pipeline.py:248), [`pipeline.py:255`](pipelines/market_snapshot/outliers/pipeline.py:255)). 테스트 9·10 은 **상대 편차를 ~10% 로 동일하게 두고 절대 편차만 가드 양쪽 끝으로 갈라** 두 분기를 검증해야 한다. 아래 조합으로 고정한다:

| 테스트 | `area` | `baseline_ppm` | `new_level = baseline × 1.30` | `ref_total_expected` (= new_level × area, 만원) | target delta | 상대 편차 | 절대 편차 | 예상 결과 |
|---|---|---|---|---|---|---|---|---|
| 9 (가드 밖, 여전히 이상치) | 100 | 약 462 | **600** | **60,000** | **+6,000** | 10% | 6,000 만원 | `is_outlier=True`, `renovation_buffer_applied=False` |
| 10 (가드 안, 완화) | 100 | 약 308 | **400** | **40,000** | **+4,000** | 10% | 4,000 만원 | `is_outlier=False` (완화), `renovation_buffer_applied=True` |

- `area=100` 을 공통으로 고정해 단위당 노이즈 흡수 + 계산이 결정적.
- `new_level` 을 테스트별로 `400` / `600` (만원/㎡) 으로 분리해 "상대 10%" 가 정확히 `4,000` / `6,000` 만원이 되도록 한다. `baseline_ppm = new_level / 1.30` (각각 약 `308` / `462`) 은 반올림 대신 float 로 주입해 breakout 비율을 정확히 유지.
- test 11 (case C 회귀) 은 test 10 의 조합을 그대로 쓰되 target 월을 `2010-01` 로 바꿔 면제 조건을 만족시킨다 — `baseline` 월들도 그에 맞춰 `2009-07 ~ 2009-12` 로 backshift. (START_YM="201001" 이면 target 월 `2010-01` 자체가 "첫 월" 이므로 면제.)

#### 체크포인트 — 내부 함수 직접 호출로 검증 [P2]

**전제**: `build_snapshot_outliers()` 는 `evaluated` 를 반환하지 않고 ([`pipelines/market_snapshot/outliers/pipeline.py:26`](pipelines/market_snapshot/outliers/pipeline.py:26)), renovation 으로 해제된 행은 최종 필터 ([`:296`](pipelines/market_snapshot/outliers/pipeline.py:296)) 에서 `outliers_df` 에서 사라진다. 따라서 `build_snapshot_outliers()` 의 공개 반환값(`outliers_df`, `market_price_df`) 만으로는 "정말 trend 경로로 진입했는지" 를 직접 검증할 수 없다.

**결정**: renovation 테스트(9~11)는 **파이프라인 내부 함수를 직접 호출**해 중간 프레임을 검증한다. 다음 두 레이어를 병행한다:

**레이어 A — 내부 함수 직접 호출 (중간 상태 검증)**:

```python
from pipelines.market_snapshot.outliers.trend_band import (
    _compute_monthly_band_frame,
    _annotate_trend_confirmation,
)

# fixture 에서 trade_df 준비 → cohort/spread/dynamic_band 를 통해 ref_price_m2, band_width_abs 가 부착된
# DataFrame (evaluated_pre_stage3 혹은 유사) 을 얻은 뒤:
monthly = _compute_monthly_band_frame(evaluated_pre_trend)
annotated = _annotate_trend_confirmation(monthly)
# [P3] merge key는 _compute_monthly_band_frame 의 group_cols 와 완전히 일치해야 한다
# (단지×면적×월). 현재 fixture 가 단일 area_repr 만 쓰더라도 key 를 고정해 재사용 안전성 확보.
merged = evaluated_pre_trend.merge(
    annotated[[
        "aptSeq", "area_repr", "month",
        "trend_confirmed", "month_price_m2", "month_row_band_abs", "month_trade_count",
    ]],
    on=["aptSeq", "area_repr", "month"], how="left",
)

target_row = merged[(merged["aptSeq"] == "B") & (merged["month"] == target_month) & (merged["is_target"])].iloc[0]
assert target_row["trend_confirmed"] == True
assert target_row["month_trade_count"] >= 3
assert abs(target_row["price_per_m2"] - target_row["month_price_m2"]) > target_row["month_row_band_abs"]
# [P2] renovation 게이트는 spread ref_m2 (= ref_price_m2 컬럼) 기준으로 dev_pct_arr 를 계산한다
# ([`pipeline.py:248`](pipelines/market_snapshot/outliers/pipeline.py:248),
#  [`pipeline.py:261`](pipelines/market_snapshot/outliers/pipeline.py:261)).
# 최종 `price_deviation_pct` 는 trend-row outlier 에서 `month_price_m2` 기준으로 덮어쓰여
# renovation 진입 시점의 값과 다르고, 또한 `evaluated_pre_trend` 단계에는 아직 붙어 있지 않을 수 있다.
# 따라서 체크포인트는 spread ref 기준으로 직접 계산한다.
rel_dev_spread = abs(
    (target_row["price_per_m2"] - target_row["ref_price_m2"])
    / target_row["ref_price_m2"]
    * 100.0
)
assert rel_dev_spread <= 12.0                                # renovation REL_CAP (0.12) 안쪽
assert target_row["age"] >= 20
```

"cohort/spread/dynamic_band 까지만 돌린 프레임" 을 얻는 경로가 `build_snapshot_outliers` 내부 블록이므로, 테스트 유틸 `_build_pre_trend_frame(trade_df)` 를 테스트 파일 안에 helper 로 두고 파이프라인의 해당 구간을 그대로 복사하거나, `build_snapshot_outliers` 를 소규모 리팩터(내부 단계 함수 추출) 해 재사용하는 방식 중 **전자(helper 복사)** 를 기본으로 한다 (프로덕션 코드 변경 최소화). 리팩터가 필요해지면 후속 이슈로 분리.

**레이어 B — 공개 반환값 검증 (계약 검증)**:

9·10 에서는 `build_snapshot_outliers()` 도 함께 호출해 최종 계약을 본다:

- 9: `outliers_df` 에 target 행이 **남아 있음** (`renovation_buffer_applied=False`, `is_outlier=True`) — 6천만 원 가드 초과로 완화되지 않았음.
- 10: `outliers_df` 에 target 행이 **없음**; `market_price_df.renovation_buffer_count.sum() >= 1` 로 KPI 반영 확인.
- 11: `outliers_df` 에 target 행이 **없음** (면제); `market_price_df.renovation_buffer_count` 에 **계수 안 됨** (case C 플래그 정리 확인).

레이어 A 가 "진입 경로가 의도대로 trend 인지", 레이어 B 가 "최종 동작이 스펙과 맞는지" 를 각각 담보한다.

대안으로 `sanity_error` 경로 (log-ratio > ln 2) 를 쓰면 is_outlier=True 는 간단히 만들 수 있지만, sanity 는 renovation buffer 대상이 아닌 방향(극단 오류 의심)이므로 의미가 맞지 않는다. **trend_month_robust_band 경로가 정답**.

### 로그 검증 방식 [P1]

본 저장소는 [`loguru`](pipelines/market_snapshot/outliers/pipeline.py:7) 를 직접 사용하며 [`pyproject.toml`](pyproject.toml) 에 `pytest-loguru` 같은 `caplog` 브리지가 설정되어 있지 않다. pytest 내장 `caplog` 는 표준 `logging` 만 가로채므로 `logger.info(...)` 는 잡히지 않는다.

**선택지**:

1. **테스트 전용 loguru sink 추가** (권장) — fixture 에서 `loguru.logger.add(list.append, ...)` 로 레코드를 리스트에 수집 후 검증, teardown 에서 `logger.remove(handle)`.
2. **`logger.info` monkeypatch** — `monkeypatch.setattr(pipeline_module.logger, "info", recorder)` 로 호출 인자 캡처.

**결정**: (1) loguru sink fixture 방식. 공용 fixture 로 만들어 테스트 11 등에서 재사용.

**fixture 위치 잠금 [P3]**: **`tests/test_market_snapshot_pipeline.py` 내부의 module-scope fixture** 로 둔다 (새 `conftest.py` 를 만들지 않는다). 이유:

- 현재 [`tests/`](tests) 디렉터리에 `conftest.py` 가 없고 ([`tests/conftest.py`](tests/conftest.py) 부재), 이 fixture 는 본 파일에서만 쓰인다. 단일 파일 로컬 fixture 가 가장 적은 파일 변경으로 끝난다.
- 향후 다른 테스트 파일이 같은 sink 를 필요로 하면 그 시점에 `tests/conftest.py` 로 승격한다 (YAGNI).

```python
# tests/test_market_snapshot_pipeline.py 상단 (기존 import 블록 근처)
import pytest
from loguru import logger

@pytest.fixture
def loguru_records():
    records: list[str] = []
    handle = logger.add(records.append, level="INFO", format="{message}")
    yield records
    logger.remove(handle)
```

테스트 11 에서는 `"renovation_buffer_raw=1" in "\n".join(records)` 등으로 간단 검증. `caplog` 는 사용하지 않는다.
12. **`test_missing_age_column_does_not_crash`** — 입력 DataFrame 에 `age` 컬럼을 아예 제거한 상태로 파이프라인 호출 → 예외 없이 완료, 면제 대상이 모두 `age_zero_exempt` 에서 제외됨.

## 롤아웃 / 후속 작업

- 파이프라인 재실행: `uv run python scripts/run_full_pipeline.py` (또는 `build_summary.py`).
- 대시보드 캐시: `@st.cache_data` 적용 로더이므로 앱 재시작/캐시 무효화 필요.
- 하위 호환: `outlier_reason` 에 새 값이 추가되는 것 외 스키마 변경 없음. 구 parquet 과 혼용 시 `A3_REASON_LABELS.get(key, key)` 로 폴백되어 무해.

### 후속 이슈 (본 PR 스코프 밖)

- `exempt_reason` 진단 출력이 필요해질 경우 별도 `snapshot_outlier_exempt.parquet` 추가.
- `deviation_total_krw` 컬럼명 리네이밍 (`deviation_total_manwon` 등). 본 PR 은 표시 로직만 수정.

## 예상 범위

- 설정: 1상수 추가(`ABS_DEVIATION_MANWON`) + 1상수 교체(`RENOVATION_ABS_BUFFER_KRW` → `RENOVATION_ABS_BUFFER_MANWON`). `FIRST_SNAPSHOT_PERIOD` 는 함수 내부 파생으로 상수화하지 않음.
- 파이프라인 ~35줄 + import 전파 1곳, 대시보드 라벨/색/편차 표시 4줄, 테스트 12케이스 + 헬퍼 `start_month` / `age` / `price_manwon` 파라미터화 + `loguru_records` fixture.
- 파이프라인 런타임 영향 미미 (마스크 2~3개 추가 연산).
- **renovation 버그 수정으로 인해 기존 이상치/시세 결과 수치가 소폭 변한다** (무력화되어 있던 절대 가드 복원). 롤아웃 시 데이터 비교 리포트 1회 수행 권장.
