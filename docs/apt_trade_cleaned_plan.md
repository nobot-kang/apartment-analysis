# `cleaned_apt_trade` 데이터셋 생성 계획 (v6 — 리뷰 반영판)

## 변경 이력

- **v1 → v2**: 산출물 prefix 변경(`cleaned_apt_trade_*`), `build_snapshot_outliers` 노출 계약 축소, `row_id` partition completeness, variant 분리, atomic write, dtype 최적화 Phase 분리, funnel 대시보드 추가, cutover follow-up PR 분리.
- **v5 → v6**: 리뷰 반영. 문서 일관성 + 테스트 보강:
  - **변경 2 step 6 문구 정정** — partition assertion 실패 시 "기존 `_SUCCESS` 는 이미 삭제된 상태" 라는 v4 잔존 문구를 제거. v5 의 "`_SUCCESS` 삭제는 첫 `replace()` 직전" 정책과 일치하게, preflight/assertion 실패는 기존 validated dataset 을 그대로 유지함을 명시.
  - **manifest `files[].path` 기준 고정** — `data/` 기준 상대경로. 예: `processed/cleaned_apt_trade_2010.parquet`, `preprocessed_plus/cleaned_trade_yearly_summary.parquet`. summary 2종도 manifest 에 포함.
  - **변경 6 overlap 분해 단위 테스트 추가** — `tests/test_data_preprocessing.py` 에 raw fixture (cancel only / direct only / cancel+direct / neither 4종) 로 신규 4개 컬럼과 `removed_cancel` / `removed_direct` 귀속 규칙까지 검증.
  - **변경 7 empty-state 중복 제거** — manifest-gated 버전만 남기고 generic 버전 삭제.
- **v4 → v5**: 리뷰 반영. 핵심 변경:
  - **`_SUCCESS` 삭제 시점 지연** — "빌드 시작 직후" → "모든 preflight + assertion 통과 + 모든 `*.tmp` 작성 완료 후, 첫 `replace()` 직전". preflight 단계 실패 시 직전 validated dataset 의 availability 보존.
  - **cancel/direct overlap 귀속 규칙 잠금** — preprocessing 순서(cancel→direct)에 맞춰 `removed_cancel = cancel_only + cancel_and_direct`, `removed_direct = direct_only`. funnel/page 15 그래프 합산이 항상 100% 매치.
  - **page 15 의 summary 소스 명시 join** — `trade_filter_yearly_summary` + `cleaned_trade_yearly_summary` + `cleaned_trade_outlier_reason_summary` 3-way join. cleaned 본문 청크는 로드하지 않음.
  - **로더 분리** — 가벼운 `get_cleaned_trade_manifest()` (validity 확인만) 와 본문 로드용 `load_cleaned_trade()` 를 분리. page 15 는 manifest 만 사용.
  - 테스트 8/8b 문구 정리 — v4 설계(`size_bytes + mtime_ns + sha256` + `_SUCCESS` 존재) 와 일치하게 수정.
- **v3 → v4**: 리뷰 반영. 핵심 변경:
  - **`floor1` precedence 버그 수정**: A-3 와 동일 순서(dropna → floor1) 를 명시 마스크로 표현. set 차집합 방식이 `floor==1 & 키 결측` 행을 잘못 분류하던 문제 해결.
  - **`_SUCCESS` 를 validity marker 로 격상**: replace 단계 직전에 삭제하고 모든 replace 후에만 다시 기록. manifest 에 파일 목록 + size + mtime_ns + sha256 까지 포함. 로더는 manifest 검증 통과 시에만 데이터셋 로드.
  - **변경 6 source 정정**: `_build_trade_filter_yearly_summary` 에 `cancel_or_direct_count` + `direct_only_count` + `cancel_only_count` + `after_cancel_direct_count` 직접 저장. 차감식이 아닌 직접 카운트 사용.
  - **테스트 항등식 수정**: cumulative stage count 의 합이 아니라 **transition 관계**를 검증.
  - **`--skip-aggregation` 의 부작용 문서화**: 현재 [`run_full_pipeline.py`](scripts/run_full_pipeline.py:246) 가 `run_preprocessing()` 도 함께 게이팅하므로 cleaned 도 같이 비활성화됨. 독립 실행은 `scripts/build_cleaned_trade.py`.
  - 검토 포인트 1·2 잠금: explicit_excluded 제외 정책 확정 (+ 향후 audit 샘플 산출물 옵션). funnel 시각화는 sankey 제거하고 funnel/waterfall + stacked area + reason breakdown 조합으로 확정.
- **v2 → v3**: 리뷰 반영. 핵심 변경:
  - **atomicity 표현 정정**: "run-wide atomic" 주장 철회. **per-file atomic write + 마지막 `_SUCCESS` manifest** 로 표현 정리. 진짜 dataset-wide atomicity 는 versioned dir + pointer 가 필요해 scope 외.
  - **funnel summary 의 raw stage source 명시**: `raw` / `after_cancel_direct` 는 기존 [`trade_filter_yearly_summary.parquet`](data/preprocessed_plus/trade_filter_yearly_summary.parquet) 를 join, `after_explicit_excluded` / `cleaned` 는 본 파이프라인이 계산.
  - **glob hardening 범위 확장**: [`analysis/common.py:191`](analysis/common.py:191) (`_read_chunked_dataset`) 도 같이 좁힘.
  - `floor1_ids` 판정을 A-3 와 동일한 `pd.to_numeric(..., errors="coerce").eq(1)` 로 통일.
  - idempotency 테스트는 raw file hash 가 아니라 로드한 DataFrame **content + schema** 동일성으로 검증.
  - `build_snapshot_outliers` 가 `row_id` 부재 시 함수 초입에서 private 으로 자동 생성 (외부 caller KeyError 방지). cleaned 빌더는 여전히 명시 부여.
  - `run_full_pipeline.py` 통합 위치 확정: `MarketSnapshotPipeline` 은 **추가하지 않음**. `run_cleaned_trade()` 만 [`run_preprocessing()`](scripts/run_full_pipeline.py:138) 직후에 호출 (의존성이 preprocessing output 이라).
  - 실패 정책: **로컬 기본 warning, `--strict-cleaned-trade` 또는 `CI` 환경변수 시 fatal**. warning-only 시 page 15 가 missing artifact empty state(`st.info(...)`) 를 표시하도록 의무화.
  - page 15 등록 위치 확정: `📍 시장 한눈에 보기` 그룹, page 14 (취소·직거래 비율 진단) 옆.
  - variant 정책: `data/processed/variants/` 하위 + gitignore (확정).
  - dtype Phase 2 (`category`) 시점 확정: cutover 안정화 **이후** 별도 PR.

## 배경 / 목표

후속 분석(가격 모델링, 패널 회귀, 코호트 분석 등)은 매번 raw 매매 데이터에서 취소·직거래·이상치를 다시 걸러낸다. 단일한 **정제 매매 데이터셋(`cleaned_apt_trade`)** 을 미리 만들어 두면:

- 분석 코드가 단일 진입점에서 깨끗한 데이터를 사용한다.
- 기준이 한 곳에서 관리되므로 분석 간 일관성이 보장된다.
- 페이지/노트북 기동 시간이 줄어든다.

본 계획은 **이미 사용 중인 두 가지 정제 로직을 그대로 재사용**해 `cleaned_apt_trade` 를 생성하는 파이프라인을 추가한다 — 새 판정 기준을 만들지 않는다. 본 PR 의 산출물은 새 데이터셋 자체이고, 기존 분석 코드의 입력 전환은 별도 follow-up PR (아래 "다운스트림 cutover" 절) 에서 수행한다.

## 현재 활용 중인 정제 로직 — 사실 확인

### (1) 취소·직거래 제거 — `pipelines/data_preprocessing.py`

[`DataPreprocessor.preprocess_trade()`](pipelines/data_preprocessing.py:307) 가 이미 두 단계를 적용한다.

- `_is_cancel_trade(cdealType)`: `cdealType.strip() == "O"` 인 행 제거 ([`pipelines/data_preprocessing.py:83`](pipelines/data_preprocessing.py:83), 호출부 [`:328`](pipelines/data_preprocessing.py:328))
- `_is_direct_trade(dealingGbn)`: `dealingGbn.strip() == "직거래"` 인 행 제거 ([`pipelines/data_preprocessing.py:87`](pipelines/data_preprocessing.py:87), 호출부 [`:335`](pipelines/data_preprocessing.py:335))

→ 결과는 `data/processed/apt_trade_{YYYY}.parquet` 로 저장된다. `cleaned_apt_trade` 는 이 산출물을 입력으로 삼는다 — raw 부터 다시 돌리지 않는다.

부수 산출물: 연도·지역별 비율 요약 `data/preprocessed_plus/trade_filter_yearly_summary.parquet` ([`:91`](pipelines/data_preprocessing.py:91)).

### (2) A-3 이상거래 탐지 — `pipelines/market_snapshot/outliers/pipeline.py`

[`build_snapshot_outliers(trade_df)`](pipelines/market_snapshot/outliers/pipeline.py:27) 가 다음 단계로 `is_outlier` 플래그를 부여한다.

- **Stage 0a (explicit_excluded)**: `date / price_per_m2 / area_repr / aptSeq` 결측행을 dropna ([`:44`](pipelines/market_snapshot/outliers/pipeline.py:44)).
- **Stage 0b (excluded from evaluation)**: `floor == 1` 행을 평가에서 제외 ([`:47-50`](pipelines/market_snapshot/outliers/pipeline.py:47)). 이상치도 비-이상치도 아닌 상태.
- **Stage 1 — sanity_error** ([`:92-97`](pipelines/market_snapshot/outliers/pipeline.py:92))
- **Stage 2 — band candidate + Stage 3 support → unsupported_jump** ([`:99-193`](pipelines/market_snapshot/outliers/pipeline.py:99))
- **Stage 2b — abs_deviation** (3억) ([`:195-205`](pipelines/market_snapshot/outliers/pipeline.py:195))
- **Legacy trend-row band — trend_month_robust_band** ([`:215-235`](pipelines/market_snapshot/outliers/pipeline.py:215))
- **Stage 4 — renovation buffer** ([`:259-279`](pipelines/market_snapshot/outliers/pipeline.py:259))
- **Stage 5 — exempt conditions** (첫 스냅샷 월, `age == 0`) ([`:281-302`](pipelines/market_snapshot/outliers/pipeline.py:281))

행 단위 `is_outlier` 플래그가 붙은 evaluated DataFrame 은 함수 안에서만 존재 — 외부에서 row 단위 판정을 받으려면 좁은 계약을 하나 추가해야 한다 (변경 1 참조).

## 행 분류 정책 — 정제 단계별 처리

| 단계 | 분류 | cleaned 포함? | 사유 |
|---|---|---|---|
| `cdealType == "O"` | cancelled | ✗ | 이미 `apt_trade_*` 단계에서 제거됨 |
| `dealingGbn == "직거래"` | direct_deal | ✗ | 이미 `apt_trade_*` 단계에서 제거됨 |
| A-3 dropna (`date / price_per_m2 / area_repr / aptSeq` 결측) | **explicit_excluded** | **✗** | 분석에 필요한 키가 없음. 본 계획은 카운트만 추적하고 cleaned 에서 제외 |
| `floor == 1` (정상) | floor1_unevaluated | ✓ | 이상치 평가에서 제외되었을 뿐, 이상거래가 아님 |
| `is_outlier == True` | **outlier** | ✗ | A-3 가 이상치 판정 |
| 위 어디에도 해당 안 됨 | **clean** | ✓ | 본 데이터셋의 본체 |

검증 항등식: `|input_after_cancel_direct| == |cleaned| + |outliers| + |explicit_excluded|` (`floor1_unevaluated` 는 cleaned 에 흡수됨).

## 구현 변경 — 최소 침습 + idempotency 보장

### 변경 1 — `build_snapshot_outliers` 가 row 단위 verdict 를 노출 (좁은 계약)

`pipelines/market_snapshot/outliers/pipeline.py`:

```python
def build_snapshot_outliers(
    trade_df: pd.DataFrame,
    return_verdicts: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if return_verdicts and "row_id" not in trade_df.columns:
        # fallback: future caller 가 row_id 없이 호출해도 KeyError 안 나게
        trade_df = trade_df.assign(row_id=np.arange(len(trade_df), dtype=np.int64))
    ...
    if return_verdicts:
        verdicts = evaluated[["row_id", "is_outlier", "outlier_reason"]].copy()
        return outliers_df, market_price_df, verdicts
    return outliers_df, market_price_df
```

**계약 범위**:
- 노출은 `row_id`(int64), `is_outlier`(bool), `outlier_reason`(str) 3개 컬럼만. 내부의 `evaluated` 전체는 노출하지 않는다 — 향후 A-3 리팩터 자유도 보존.
- **`row_id` 는 호출자가 명시 부여하는 것이 권장 경로**. cleaned 빌더(변경 2)는 항상 명시 부여한다. 함수 내부의 자동 생성은 future caller 의 KeyError 방지용 fallback 일 뿐, 의미상으로는 호출자 책임.
- `floor == 1` / `dropna` 로 평가에서 빠진 행은 verdicts 에도 빠져 있다 — cleaned 빌더에서 row_id 차집합으로 explicit_excluded / floor1 을 식별한다.

기본 시그니처(`return_verdicts=False`)는 그대로 두어 [`runner.py:92`](pipelines/market_snapshot/runner.py:92) 등 기존 호출부는 무영향.

### 변경 2 — 신규 모듈 `pipelines/cleaned_trade_pipeline.py`

책임 (순서):

1. **입력 로드**: `data/processed/apt_trade_{YYYY}.parquet` 를 직접 로드. `pipelines/market_snapshot/io.py:_load_all_trade` 는 사용하지 않는다 — 본 모듈은 자체 글롭 (`apt_trade_[0-9][0-9][0-9][0-9].parquet` 정규식) 으로 cleaned 산출물 재흡수를 원천 차단.
2. **`row_id` 부여**: `df.reset_index().rename(columns={"index": "row_id"})` (단조증가 정수). 본 PR 안에서만 사용하는 ephemeral key.
3. **공통 전처리**: `_add_region_columns`, `_add_area_bucket`, `_add_month_column`.
4. **이상치 평가**: `outliers_df, market_price_df, verdicts = build_snapshot_outliers(trade_df, return_verdicts=True)`.
5. **분류 — A-3 와 동일 precedence (v4 수정)**:

   A-3 는 [`pipeline.py:44`](pipelines/market_snapshot/outliers/pipeline.py:44) 에서 먼저 `date / price_per_m2 / area_repr / aptSeq` 결측을 dropna 하고, [`pipeline.py:47-50`](pipelines/market_snapshot/outliers/pipeline.py:47) 에서 그 다음 `floor == 1` 을 제외한다. 분류도 **반드시 같은 순서**로 표현해야 한다 — 그렇지 않으면 `floor==1 & 키 결측` 행이 `floor1_unevaluated` 로 잘못 분류되어 cleaned 에 포함된다.

   set 차집합 대신 **명시 마스크**로 작성한다:

   ```python
   required_keys = ["date", "price_per_m2", "area_repr", "aptSeq"]
   required_key_mask = trade_df[required_keys].notna().all(axis=1)

   floor_eq_1 = pd.to_numeric(trade_df["floor"], errors="coerce").eq(1)

   explicit_excluded_mask = ~required_key_mask                 # dropna 가 먼저
   floor1_mask            = required_key_mask & floor_eq_1     # 그 다음 floor1
   evaluated_mask         = required_key_mask & ~floor_eq_1    # A-3 가 실제로 평가한 행

   explicit_excluded_ids = trade_df.loc[explicit_excluded_mask, "row_id"]
   floor1_ids            = trade_df.loc[floor1_mask, "row_id"]
   evaluated_ids         = trade_df.loc[evaluated_mask, "row_id"]
   outlier_ids           = verdicts.loc[verdicts.is_outlier, "row_id"]
   ```

   `floor` 의 캐스팅은 dtype 최적화 **전** 단계에서 수행 (Int16 변환 후에도 동일 결과지만, 의미상 평가 시점과 일치시킨다).
6. **partition completeness 검증** (assert):
   - 마스크 disjoint: `(explicit_excluded_mask & floor1_mask).sum() == 0`, `(explicit_excluded_mask & evaluated_mask).sum() == 0`, `(floor1_mask & evaluated_mask).sum() == 0`.
   - 마스크 cover: `explicit_excluded_mask | floor1_mask | evaluated_mask` 가 모든 행 True.
   - 분류: `set(input_ids) == set(cleaned_ids) ∪ set(outlier_ids) ∪ set(explicit_excluded_ids)`, `cleaned_ids ∩ outlier_ids == ∅`, `cleaned_ids ∩ explicit_excluded_ids == ∅`.
   - 어긋나면 `RuntimeError`. 이 시점은 preflight 단계(변경 4 step 1·2)이므로 **기존 `_SUCCESS` 는 아직 삭제되지 않은 상태** — canonical 파일과 manifest 모두 무손상으로 보존되어 직전 validated dataset 을 계속 신뢰할 수 있다. `_SUCCESS` 삭제는 모든 preflight 통과 + `*.tmp` 작성 완료 후 첫 `replace()` 직전에만 일어난다(변경 4 step 3 참조).
7. **cleaned 구성**: `cleaned = trade_df[~trade_df.row_id.isin(outlier_ids ∪ explicit_excluded_ids)].drop(columns=["row_id"])`. floor1 정상 거래(`floor1_mask`) 는 자동 포함.
8. **컬럼 정리**: `apt_trade_*.parquet` 의 원본 컬럼 셋을 그대로 유지. A-3 진단 컬럼(`spread_*`, `band_pct`, `ref_price_*` 등)은 canonical artifact 에 **포함하지 않는다**. 별도 옵션이 필요하면 별도 산출물(아래 변경 5).
9. **dtype 최적화** (변경 3) 적용.
10. **Atomic 저장** (변경 4 정책):
    - 연도별 청크: `data/processed/cleaned_apt_trade_{YYYY}.parquet`
    - 통합본: `data/processed/cleaned_apt_trade.parquet` (≈80MB 예상 → gitignore)
11. **요약 산출** (변경 6).

> **명시적 비-목표**: 본 모듈은 `aggregation_pipeline.py` 의 `_load_processed_chunks("apt_trade", ...)` 같은 기존 소비자를 cleaned 로 갈아끼우지 않는다. cutover 는 별도 PR.

### 변경 3 — dtype 최적화 (cleaned 저장 직전 일괄 적용)

현재 `apt_trade_*.parquet` 의 dtype 은 전처리에서 명시적으로 좁히지 않아 대부분 64-bit 이다. 실제 값 범위를 보면 더 작은 타입으로 충분하다 (값은 `data/processed/apt_trade_2024.parquet`, 148,761 행 기준 측정).

#### Phase 1 — 본 PR 에서 적용 (안전 범위)

| 컬럼 | 현재 | Phase 1 | 근거 |
|---|---|---|---|
| `price` | `int64` | `int32` | 만원 단위. int32 max ≈ 21.4억 → 만원 환산 21조 원 |
| `price_std84` | `Int64` | `Int32` | 동일. nullable 유지 |
| `price_per_m2` | `float64` | `float32` | 만원/㎡, 소수 2자리. float32 정밀도 충분 |
| `price_per_py` | `float64` | `float32` | 동일 |
| `area` | `float64` | `float32` | ㎡, 소수 4자리. float32 충분 |
| `floor` | `float64` (NaN 포함) | `Int16` (nullable) | 정수 확정 + NaN 가능. Int8 은 빠듯 |
| `construction_year` | `int64` | `Int16` | 1900~2100 범위 |
| `age` | `float64` (NaN 포함) | `Int16` (nullable) | 정수 + 음수(`year < buildYear`) 가능성 → Int16 안전 |
| `area_repr` | `Int64` | `Int16` | 면적 평수 정수 |
| `date` | `datetime64[ns]` | `datetime64[ms]` | 일 단위 거래에 ns 불필요 |

#### Phase 2 — 별도 PR 에서 검토 (canonical schema 변경, 다운스트림 영향 큼)

| 컬럼 | Phase 2 후보 | 근거 / 보류 사유 |
|---|---|---|
| `aptSeq` | `category` | 14× 재사용 → 메모리 큼. 단 비교/머지 호환성 확인 필요 |
| `apt_name` | `category` | 17× 재사용 |
| `apt_name_repr` | `category` | 14× 재사용 |
| `dong` | `category` | 211× 재사용. 효과 큼 |
| `dong_repr` | `category` | 동일 |

리뷰 의견대로 첫 도입 PR 에서 canonical schema(특히 문자열 → category) 까지 바꾸는 건 보수적이지 않다. 본 PR 은 numeric downcast 만 적용하고, category 변환은 cutover PR 또는 그 이후로 분리한다.

#### 효과 추정 (2024 청크, 148K 행 in-memory, Phase 1만)

- 변경 전 ~32 MB → Phase 1 후 약 12–15 MB (≈2.5× 감소). 디스크 parquet 은 압축으로 약 1.3–1.5× 절감 예상.
- Phase 2 까지 적용 시 in-memory 6–9 MB 까지 추가 감소 가능.

#### 구현 헬퍼

```python
_PHASE1_DTYPES: dict[str, str] = {
    "price": "int32",
    "price_std84": "Int32",
    "price_per_m2": "float32",
    "price_per_py": "float32",
    "area": "float32",
    "floor": "Int16",
    "construction_year": "Int16",
    "age": "Int16",
    "area_repr": "Int16",
}

def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in _PHASE1_DTYPES.items():
        if col not in out.columns:
            continue
        if dtype.startswith("Int"):
            out[col] = out[col].round().astype(dtype)
        else:
            out[col] = out[col].astype(dtype)
    if "date" in out.columns:
        out["date"] = out["date"].astype("datetime64[ms]")
    return out
```

호출 위치: cleaned DataFrame 을 청크/통합본으로 저장하기 **직전**. 평가 단계에서는 원본 dtype 유지.

#### 사이드 이펙트 / 주의

- `age` Int16 변환: `buildYear==0` 일 때 큰 음수 가능 → 전처리에서 이미 `age=NaN` 처리됨([`pipelines/data_preprocessing.py:203`](pipelines/data_preprocessing.py:203)). 안전.
- `floor==NaN` 처리: A-3 가 `pd.to_numeric(..., errors="coerce")` 로 처리하므로 nullable Int 도 안전 ([`pipelines/market_snapshot/outliers/pipeline.py:47`](pipelines/market_snapshot/outliers/pipeline.py:47)).
- 검증: 테스트에 dtype 어서션 추가 — `assert cleaned["floor"].dtype == "Int16"` 등.

### 변경 4 — Per-file atomic write + `_SUCCESS` validity marker (v4)

> **표현 정정 (v3)**: 본 PR 은 "run-wide atomic" 을 약속하지 않는다. flat-file 산출물에서 dataset-wide atomicity 를 보장하려면 versioned output dir + pointer 구조가 필요하고, 이는 본 PR 의 scope 밖이다.
>
> **v4 보강 + v5 정정**: `_SUCCESS` 를 **단순 timestamp 기록이 아니라 데이터셋 유효성 marker** 로 격상한다. 핵심 규약:
>
> 1. **`_SUCCESS` 삭제는 "첫 `replace()` 직전" 까지 지연한다 (v5 정정)** — 모든 preflight (input load, partition assertion, summary 산출) + 모든 `*.tmp` 작성이 통과한 직후 + 첫 `replace()` 호출 직전 시점. 그 이전 단계 실패는 canonical 파일을 건드리지 않으므로, 이전 validated dataset 을 계속 신뢰할 수 있다 (availability 보존).
> 2. **모든 `replace()` 가 끝난 뒤에만 `_SUCCESS` 를 다시 기록** 한다.
> 3. `_SUCCESS` 안의 manifest 는 파일 목록 + 각 파일의 `size_bytes` + `mtime_ns` + `sha256` 을 포함.
> 4. 로더는 `_SUCCESS` 가 존재하고 manifest 의 모든 파일이 size+mtime(+옵션 sha256) 일치할 때만 데이터셋을 신뢰한다.
>
> 효과: "이전 `_SUCCESS` + 일부 새 파일 + 일부 옛 파일" 의 혼합 상태가 생겨도, 로더는 manifest 검증에 실패해 데이터셋을 거절한다. **preflight 단계 실패는 가용성에 영향 없음**, replace 단계 중 실패만 데이터셋 invalidation 으로 이어진다.

#### 저장 절차 (v4)

```python
def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)  # POSIX rename: atomic on same filesystem (per-file only)
```

전체 절차 (v5 순서):

1. **Preflight** — input 로드, partition completeness assertion (변경 2 step 6) 통과.
2. **모든 산출물(연도 청크, 통합본, 요약 2종)을 `*.tmp` 로 작성** — 이 단계 실패해도 canonical 파일·`_SUCCESS` 무손상.
3. **`_SUCCESS` unlink** (있으면) — 첫 `replace()` 직전. 이 시점부터 데이터셋은 invalid 상태로 표시되며, 다음 단계가 일부라도 실패하면 그대로 남는다.
4. **모든 `*.tmp` 를 일괄 `replace()`** (per-file atomic).
5. **manifest 계산 + `_SUCCESS` 기록**:

   **`files[].path` 기준 (v6 잠금)**: 모든 path 는 **리포 루트의 `data/` 를 기준으로 한 상대경로** 로 기록한다. POSIX 구분자(`/`) 고정. 검증 대상에는 연도 청크 + 통합본 + summary 2종 전부 포함한다. `_SUCCESS` 자체는 manifest 에 등재하지 않는다 (자기 자신을 검증할 수 없으므로).

   ```json
   {
     "build_ts": "2026-04-23T12:34:56Z",
     "git_rev": "abc1234",
     "row_counts": {"input": 2_457_812, "cleaned": 2_389_104, "outlier": 41_203, "explicit_excluded": 27_505},
     "files": [
       {
         "path": "processed/cleaned_apt_trade_2010.parquet",
         "size_bytes": 3_123_456,
         "mtime_ns": 1745420496123456789,
         "sha256": "..."
       },
       {
         "path": "processed/cleaned_apt_trade.parquet",
         "size_bytes": 83_456_789,
         "mtime_ns": 1745420500123456789,
         "sha256": "..."
       },
       {
         "path": "preprocessed_plus/cleaned_trade_yearly_summary.parquet",
         "size_bytes": 12_345,
         "mtime_ns": 1745420501123456789,
         "sha256": "..."
       },
       {
         "path": "preprocessed_plus/cleaned_trade_outlier_reason_summary.parquet",
         "size_bytes": 6_789,
         "mtime_ns": 1745420502123456789,
         "sha256": "..."
       }
     ]
   }
   ```

   `get_cleaned_trade_manifest()` / `load_cleaned_trade()` 는 `data_root` (기본: 리포 루트의 `data/`) 와 `files[].path` 를 join 하여 실제 경로를 해석한다.

6. **실패 시 동작 (단계별)**:
   - Step 1·2 실패 → canonical/`_SUCCESS` 무손상. `*.tmp` 정리. 직전 dataset 그대로 사용 가능.
   - Step 3 후 ~ Step 5 전 실패 → canonical 파일은 일부 새 버전이지만 `_SUCCESS` 없음 → 로더 거절. 이 케이스만 dataset invalidation.
   - Step 5 (manifest 기록) 자체 실패 → 동일하게 `_SUCCESS` 없음 → 로더 거절.

#### 로더 측 헬퍼 — 가벼운 검증과 본문 로드 분리 (v5)

요약 페이지(page 15) 처럼 manifest validity 만 필요한 소비자가 cleaned 본문 청크를 검증·로드하는 비용을 치르지 않도록 두 함수로 분리한다.

```python
# analysis/common.py
@dataclass(frozen=True)
class CleanedTradeManifest:
    build_ts: str
    git_rev: str
    row_counts: dict[str, int]
    files: list[dict]   # [{"path", "size_bytes", "mtime_ns", "sha256"}, ...]

def get_cleaned_trade_manifest(
    data_root: Path | None = None,   # 기본: <repo>/data
    verify_files: bool = True,
    verify_sha256: bool = False,
) -> CleanedTradeManifest:
    """validity 만 확인. 청크는 절대 로드하지 않는다.
    - _SUCCESS 부재 → FileNotFoundError
    - manifest 파싱 실패 → RuntimeError
    - verify_files=True 일 때 size+mtime 불일치 → RuntimeError
    - verify_sha256=True 일 때 sha 불일치 → RuntimeError

    files[].path 는 data_root 기준 상대경로 (예: "processed/cleaned_apt_trade_2010.parquet").
    실제 검증 경로는 data_root / path 로 해석.
    """

def load_cleaned_trade(
    data_root: Path | None = None,
    columns: list[str] | None = None,
    verify_sha256: bool = False,
) -> pd.DataFrame:
    """본문 로드. 내부적으로 get_cleaned_trade_manifest() 검증을 먼저 통과한 뒤 청크 concat.
    """
```

소비자별 사용:
- **page 15 (변경 7)** → `get_cleaned_trade_manifest()` 만 호출 (validity gate). 통과 시 summary parquet 들만 로드, 본문 청크는 건드리지 않음. 검증 실패 시 `st.info(...)` empty state.
- **다운스트림 분석 (cutover PR)** → `load_cleaned_trade()` 사용.
- **CI 검증** → `verify_sha256=True` 옵션으로 강검증.

#### 실패 정책

- **partition completeness assertion 실패** → `RuntimeError`. 기존 산출물·manifest 무손상.
- **개별 파일 write 실패** → 위 절차로 부분 산출물 미발생. 기존 manifest 시점 신뢰 가능.
- **CLI 단독 실행 (`scripts/build_cleaned_trade.py`)** → 실패 시 비-0 종료 코드로 즉시 종료.
- **`run_full_pipeline.py` 통합 호출 시 — 환경에 따라 분기**:
  - 기본(로컬 개발) → **warning** 처리. 전체 파이프라인 비-실패. sentinel 로그 `CLEANED_PIPELINE_FAILED` 남김.
  - `--strict-cleaned-trade` 플래그 OR 환경변수 `CI=true` → **fatal**. 전체 파이프라인 실패. CI 가 잡을 수 있도록 비-0 종료.
  - 사유: cleaned 는 정기 갱신의 종속 산출물이라 로컬에서는 partial run 을 허용하는 것이 개발 friction 을 줄인다. 반면 CI 는 항상 완전한 산출물을 기대해야 한다.

### 변경 5 — CLI 진입점 (canonical artifact 의 의미는 옵션으로 바뀌지 않는다)

스크립트 `scripts/build_cleaned_trade.py`:

```bash
# canonical: 항상 같은 스키마, 같은 파일명
uv run python scripts/build_cleaned_trade.py

# variant: 별도 디렉터리·파일명 — canonical 을 덮어쓰지 않음
uv run python scripts/build_cleaned_trade.py --variant with-diagnostics
#   → data/processed/variants/cleaned_apt_trade_with_diag_{YYYY}.parquet
#   (A-3 진단 컬럼 부착)

uv run python scripts/build_cleaned_trade.py --variant exclude-floor1
#   → data/processed/variants/cleaned_apt_trade_no_floor1_{YYYY}.parquet
#   (1층 거래도 제거)

# 운영 옵션
uv run python scripts/build_cleaned_trade.py --no-combined   # 통합본 생략
uv run python scripts/build_cleaned_trade.py --years 2023 2024  # 일부 연도만
```

핵심 원칙: `cleaned_apt_trade*.parquet` (variant 접두사 없음) 은 **항상 동일한 스키마와 분류 규칙**을 가진다. 다른 의미가 필요하면 다른 파일명, 다른 디렉터리.

`scripts/run_full_pipeline.py` 통합 hook 위치 (v3 확정):
- `MarketSnapshotPipeline` 은 본 PR 에서 추가하지 **않는다** — cleaned 의 실제 의존성은 preprocessing output 이지 snapshot 이 아니다.
- `run_cleaned_trade()` 를 [`run_preprocessing()`](scripts/run_full_pipeline.py:138) 호출 직후에 추가.
- `--skip-cleaned-trade` 플래그 신설 (기존 `--skip-*` 패턴과 동일 그룹).
- `--strict-cleaned-trade` 플래그 신설 (변경 4 의 fatal/warning 정책 제어).

#### `--skip-aggregation` 의 부작용 (v4 문서화)

현재 [`run_full_pipeline.py:246-252`](scripts/run_full_pipeline.py:246) 는 `--skip-aggregation` 가 켜지면 `run_preprocessing()` **과** `run_aggregation()` **둘 다** 건너뛰는 구조다. cleaned 단계는 preprocessing 산출물에 의존하므로 자연스럽게 같이 비활성화된다. 이는 큰 blocker 는 아니지만 사용자 혼동을 막기 위해 문서에 명시한다:

- `uv run python scripts/run_full_pipeline.py --skip-aggregation` → preprocessing + aggregation + cleaned 모두 skip.
- cleaned 단독 갱신이 필요하면 **`uv run python scripts/build_cleaned_trade.py` 가 정식 경로**.
- `--skip-aggregation` 의 의미를 둘로 쪼개는 것(예: `--skip-preprocessing` 별도 신설) 은 본 PR 의 scope 밖.

### 변경 6 — 진단 산출물 (`data/preprocessed_plus/`)

`cleaned_trade_yearly_summary.parquet` — long format (스키마 안정성 확보, parquet/대시보드 친화):

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `year` | int | 연도 |
| `region_scope` | str | `"전체"` / `"서울"` / `"경기"` / sggCd 단위 |
| `sggCd` | str | sggCd (집계 행은 `"ALL"` / `"SEOUL"` / `"GYEONGGI"`) |
| `stage` | str | `"raw"` / `"after_cancel_direct"` / `"after_explicit_excluded"` / `"after_outlier"` (= cleaned) |
| `count` | int64 | 해당 stage 의 행수 |
| `pct_of_raw` | float | raw 대비 잔존 비율 |
| `removed_count` | int64 | 직전 stage 대비 제거 행수 |

#### Stage 별 source (v4 정정)

cleaned 빌더의 입력은 이미 `apt_trade_{YYYY}.parquet` (= 취소·직거래 이미 제거된 상태) 이므로 `raw` / `after_cancel_direct` 카운트를 직접 알 수 없다. 두 단계는 [`trade_filter_yearly_summary.parquet`](data/preprocessed_plus/trade_filter_yearly_summary.parquet) join 으로 채운다.

**v3 산식의 문제**: `after_cancel_direct = total − cancel − direct` 는 cancel 과 direct 가 **disjoint 일 때만** 옳다. 한 거래가 동시에 둘 다일 수 있으면(`cdealType=="O"` AND `dealingGbn=="직거래"`) 차감식은 음수 쪽으로 어긋난다. 그리고 page 15 가 `removed_cancel %` / `removed_direct %` 를 분리해 그리려면 overlap 분해가 필요하다.

**v4 해결책**: [`DataPreprocessor._build_trade_filter_yearly_summary()`](pipelines/data_preprocessing.py:91) 에 다음 컬럼을 **본 PR 안에서 추가** 한다 (raw 데이터에 한 번 더 접근 가능한 유일한 위치):

| 신규 컬럼 | 산식 | 용도 |
|---|---|---|
| `cancel_only_count` | cancel=O AND direct=False | page 15 `removed_cancel %` |
| `direct_only_count` | cancel=False AND direct=True | page 15 `removed_direct %` |
| `cancel_and_direct_count` | cancel=O AND direct=True | overlap 분해 (실측치 확인용) |
| `after_cancel_direct_count` | NOT (cancel OR direct) | summary `after_cancel_direct` 직접 source |

`cancel_trade_count` / `direct_trade_count` 는 하위호환을 위해 유지 (= `*_only + cancel_and_direct`). 기존 [`page_14_trade_filter_diagnostics`](dashboard/pages/page_14_trade_filter_diagnostics.py) 도 영향 없음.

#### Cancel/Direct overlap 귀속 규칙 (v5 잠금)

[`DataPreprocessor.preprocess_trade()`](pipelines/data_preprocessing.py:307) 의 실제 처리 순서: cancel 먼저([`:328`](pipelines/data_preprocessing.py:328)) → direct 다음([`:335`](pipelines/data_preprocessing.py:335)). 이 순서를 그대로 funnel attribution 에 반영한다 — overlap 행은 cancel 단계에서 이미 빠져 direct 단계로 넘어가지 않기 때문.

| funnel bucket | 산식 | 정당화 |
|---|---|---|
| `removed_cancel` | `cancel_only + cancel_and_direct` | cancel 단계가 먼저 두 부류 모두 제거 |
| `removed_direct` | `direct_only` | direct 단계는 남은 direct-only 만 제거 |

검증 항등식: `removed_cancel + removed_direct + after_cancel_direct_count == raw` (= `total_trade_count`).
이 규칙으로 page 15 stacked area 의 합산이 항상 100% 매치한다.

stage source 표 (v4):

| stage | count source | 비고 |
|---|---|---|
| `raw` | `trade_filter_yearly_summary.total_trade_count` | |
| `after_cancel_direct` | `trade_filter_yearly_summary.after_cancel_direct_count` | **차감식 아닌 직접 카운트** |
| `after_explicit_excluded` | cleaned pipeline 자체 | `after_cancel_direct − len(explicit_excluded_ids)` |
| `after_outlier` (= cleaned) | cleaned pipeline 자체 | `after_explicit_excluded − len(outlier_ids)` |

#### 검증 (변경 2 step 6 와 연계)

- 정합성 게이트: `after_cancel_direct_count` (요약) == 입력 `apt_trade_{YYYY}` 의 실제 행수.
- 불일치 시 `RuntimeError` — `trade_filter_yearly_summary.parquet` 가 stale 하다는 신호.
- `trade_filter_yearly_summary.parquet` 자체 부재 시 (이론상 불가 — 본 PR 에서 [`DataPreprocessor`](pipelines/data_preprocessing.py:319) 가 항상 같이 생성) 빌더는 `RuntimeError` 로 fail-fast.

#### outlier reason 별 카운트

별도 long table `cleaned_trade_outlier_reason_summary.parquet`:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `year` / `region_scope` / `sggCd` | — | 동일 |
| `outlier_reason` | str | `sanity_error` / `unsupported_jump` / `abs_deviation` / `trend_month_robust_band` |
| `count` | int64 | |

dict/object 컬럼은 사용하지 않는다.

### 변경 7 — 신규 대시보드 페이지: 정제 funnel

`dashboard/pages/page_15_cleaned_trade_funnel.py` (신규):

- **상단 funnel/waterfall (선택 연도 1개)** — sankey 는 채택하지 않음 (단계가 4 개뿐이라 가독보다 장식. 구현 비용 높음). plotly `funnel` 또는 `waterfall` trace 로 단순하게:
  - raw → after_cancel_direct → after_explicit_excluded → cleaned
  - 각 단계의 행수 + 직전 대비 제거율 라벨.
- **중단 stacked area chart (연도별 추세)**:
  - x축: year
  - y축: percent (각 연도 raw = 100%)
  - 누적 area: `cleaned %` (가장 아래) → `removed_outlier %` → `removed_explicit_excluded %` → `removed_direct %` → `removed_cancel %`
- **하단 outlier reason 분해**: 연도별 stacked bar — `sanity_error` / `unsupported_jump` / `abs_deviation` / `trend_month_robust_band` 의 비율.

#### 데이터 소스 (v5 명시 join)

page 15 의 모든 차트는 **3-way join 으로 구성된 단일 in-memory frame** 위에 그린다. cleaned 본문 청크는 로드하지 않는다.

| 컬럼 | source parquet | 비고 |
|---|---|---|
| `removed_cancel` | `trade_filter_yearly_summary.parquet` | `cancel_only_count + cancel_and_direct_count` (변경 6 잠금 규칙) |
| `removed_direct` | `trade_filter_yearly_summary.parquet` | `direct_only_count` |
| `after_cancel_direct` | `trade_filter_yearly_summary.parquet` | `after_cancel_direct_count` (정합성 게이트 키) |
| `removed_explicit_excluded` | `cleaned_trade_yearly_summary.parquet` | `after_cancel_direct − after_explicit_excluded` |
| `removed_outlier` | `cleaned_trade_yearly_summary.parquet` | `after_explicit_excluded − after_outlier` |
| `cleaned` | `cleaned_trade_yearly_summary.parquet` | `after_outlier` |
| `outlier_reason_*` | `cleaned_trade_outlier_reason_summary.parquet` | reason 별 wide pivot |

join key: `(year, region_scope, sggCd)`. region_scope 토글(전체/서울/경기)에 따라 필터링.

> **본 PR 안에서는 page 15 전용 denormalized summary 를 따로 만들지 않는다** — 3-way join 의 비용이 page 15 수준에서 무시할 만하고, summary 파일을 늘리면 재계산 트리거 점이 분산되어 manifest validity 의 의미가 약해진다.

#### Empty state (v3 의무 + v5 분리 반영)

페이지 진입 시:

1. `get_cleaned_trade_manifest()` 호출 (변경 4 헬퍼). `_SUCCESS` 부재 / 검증 실패 시 `st.info(...)` 안내 + return.
2. 통과 시 위 3-way join (summary parquet 만 로드) 수행.
3. summary parquet 자체 부재 시도 동일 empty state.

```python
try:
    get_cleaned_trade_manifest(verify_files=True)
except (FileNotFoundError, RuntimeError) as exc:
    st.info(
        "정제 매매 데이터셋이 아직 검증되지 않았습니다. "
        "`uv run python scripts/build_cleaned_trade.py` 실행 후 다시 확인해 주세요. "
        f"(상세: {exc})"
    )
    return
```

run_full_pipeline 의 cleaned 단계가 warning-only 로 실패한 경우에도 이 empty state 가 노출된다.

#### 등록 위치 (v3 확정)

[`dashboard/app.py:18`](dashboard/app.py:18) `NAVIGATION` 의 `"📍 시장 한눈에 보기"` 그룹, **`"취소·직거래 비율 진단"` (page 14) 바로 아래**에 추가. funnel 은 page 14 의 cancel/direct 단계를 **포함하면서 그 뒤 단계까지 확장**한 시각화이므로 시각적으로 인접 배치가 자연스럽다.

```python
"📍 시장 한눈에 보기": {
    ...
    "취소·직거래 비율 진단": ("dashboard.pages.page_14_trade_filter_diagnostics", ...),
    "정제 단계별 거래 잔존율": ("dashboard.pages.page_15_cleaned_trade_funnel", "render_funnel"),  # 신규
    ...
}
```

본 페이지는 새 데이터셋의 사용 사례 첫 예시 역할도 겸한다.

### 변경 8 — `.gitignore`

LFS 제한(GitHub 권장 50MB / 차단 100MB)을 넘을 가능성이 있는 산출물만 차단한다. **연도 청크는 깃 추적을 허용**(소형이고 분석 재현성에 유용).

기존 [`.gitignore`](.gitignore):

```
data/processed/apt_trade.parquet
data/processed/apt_rent.parquet
```

추가:

```
# A-3 정제 매매 데이터 — 통합본만 차단 (LFS 임계 근접). 연도 청크는 추적.
data/processed/cleaned_apt_trade.parquet

# variant 산출물은 전체 무시 (실험성)
data/processed/variants/
```

연도 청크는 ~5–10MB 이므로 차단하지 않는다. 단일 연도가 100MB 를 초과하면 빌더가 경고 로그를 남긴다 — 그 때 `cleaned_apt_trade_*.parquet` 패턴 차단으로 확장한다.

### 변경 9 — 다운스트림 cutover (별도 follow-up PR)

본 PR 의 산출물은 새 데이터셋 + 신규 funnel 페이지뿐이고, 기존 분석 코드의 입력 전환은 별도 PR 에서 한다. 대상:

| 호출부 | 현재 prefix | 변경안 |
|---|---|---|
| [`pipelines/market_snapshot/io.py:_load_all_trade`](pipelines/market_snapshot/io.py:12) | `apt_trade_*` | `apt_trade_[0-9][0-9][0-9][0-9].parquet` 정규 글롭으로 좁힘 (cleaned 흡수 차단). cutover 시 cleaned 사용 여부 검토 |
| [`pipelines/aggregation_pipeline.py:_load_processed_chunks`](pipelines/aggregation_pipeline.py:176) | `{prefix}_*.parquet` | prefix 인자 자체를 `apt_trade` 대신 `cleaned_apt_trade` 로 호출하는 쪽을 점진 전환 |
| [`analysis/common.py:_read_chunked_dataset`](analysis/common.py:191) | `apt_trade` | 동일 |
| 대시보드 detail 로더 | `apt_trade_*` | 동일 |

> **방어 조치 (v3 범위 확장)**: 본 PR 안에서 다음 **세 군데** 글롭을 **연도 자릿수 정규 글롭** (`apt_trade_[0-9][0-9][0-9][0-9].parquet`) 으로 좁힌다. 이는 cutover 와 무관하게, 다른 prefix 산출물(`cleaned_apt_trade_*` 등)이 실수로 유입되는 가능성을 0 으로 만들기 위한 idempotency 안전장치다. 호출부 의미는 그대로 유지.
>
> 1. [`pipelines/market_snapshot/io.py:14`](pipelines/market_snapshot/io.py:14) — `_load_all_trade` (rent loader 도 동일 패턴 적용)
> 2. [`pipelines/aggregation_pipeline.py:182`](pipelines/aggregation_pipeline.py:182) — `_load_processed_chunks`. prefix 인자에 따라 `{prefix}_[0-9][0-9][0-9][0-9].parquet` 형태로 일반화.
> 3. [`analysis/common.py:191`](analysis/common.py:191) — `_read_chunked_dataset`. (현재 `cleaned_apt_trade_*` 와 직접 충돌하지는 않지만, 미래 회귀 차단을 위해 같이 좁힘.)
>
> 셋의 공통 헬퍼를 따로 두지는 않는다 — 각자 인접 코드와 함께 좁히는 PR diff 가 가독성이 더 좋다.

## 산출 파일 위치 정리

| 파일 | 위치 | 깃 추적 |
|---|---|---|
| `cleaned_apt_trade_{YYYY}.parquet` | `data/processed/` | ✓ |
| `cleaned_apt_trade.parquet` (통합) | `data/processed/` | ✗ (gitignore) |
| `cleaned_trade_yearly_summary.parquet` | `data/preprocessed_plus/` | ✓ |
| `cleaned_trade_outlier_reason_summary.parquet` | `data/preprocessed_plus/` | ✓ |
| `cleaned_apt_trade_with_diag_*.parquet` 등 variants | `data/processed/variants/` | ✗ |

## 테스트 (`tests/test_cleaned_trade_pipeline.py`)

리뷰의 회귀 테스트 항목을 모두 포함한다.

1. **partition completeness**: `set(input_ids) == set(cleaned_ids) ∪ set(outlier_ids) ∪ set(explicit_excluded_ids)`, 교집합 ∅.
2. **이상치 분류 정확성**:
   - 가격 10× 이상치 → cleaned 에서 빠짐.
   - 1층 정상 거래 → cleaned 에 포함.
   - `age==0` 신축 고가 → 면제, cleaned 포함.
   - `START_YM` 첫 월 거래 → 면제, cleaned 포함.
3. **재실행 idempotency**: 두 번 실행해도 **로드한 DataFrame 의 content + schema** 가 동일. 파일 raw hash 가 아니라 `pd.testing.assert_frame_equal` (이상치/cleaned 양쪽 + summary 2종) + dtype 어서션으로 검증한다 — parquet metadata(`pandas_version`, write timestamp 등) 때문에 raw byte hash 는 brittle.
4. **glob isolation**: 디렉터리에 가짜 `cleaned_apt_trade_2099.parquet` 을 두고 빌더 실행 → 입력 통합본에 들어가지 않음.
5. **`build_snapshot_outliers` 기본 2-tuple 반환 유지** — 기존 호출부 회귀 테스트.
6. **duplicate row 보존**: 동일 키(date, aptSeq, area, price) 의 중복 행도 row_id 로 구분되어 유실되지 않음.
7. **CLI 옵션별 산출물 계약**:
   - canonical 산출물 파일명·스키마는 어떤 옵션에도 변하지 않음.
   - `--variant with-diagnostics` 는 `variants/` 하위에만 쓰고 canonical 파일은 건드리지 않음.
8. **`_SUCCESS` validity marker 동작 (v5 정정)** — 데이터셋 유효성의 단일 진실 소스는 `_SUCCESS` 의 **존재 + manifest 검증 통과** 여부.
   - **Preflight 단계 실패 (변경 4 step 1·2)** → 기존 `_SUCCESS` 그대로 존재. 직전 dataset 계속 valid (availability 보존).
   - **Replace 단계 중 실패 (변경 4 step 3 이후)** → `_SUCCESS` 부재 상태로 종료 → 로더 거절. canonical 파일이 혼합 상태여도 정상으로 오인 안 됨.
   - **정상 종료** → manifest 의 `files[]` 목록이 디스크 실제 파일의 `size_bytes` + `mtime_ns` + `sha256` 와 일치.
   - 검증: `get_cleaned_trade_manifest(verify_files=True, verify_sha256=True)` 호출이 정상 종료 후엔 통과, 강제 mtime/size 변조 후엔 `RuntimeError`.
9. **summary count transition 검증 (v4 정정)** — stage count 는 cumulative 라 단순 합산은 맞지 않다. 다음 transition 관계로 검증:
   - `after_cancel_direct == input_rows_for_apt_trade_chunk`
   - `after_explicit_excluded == after_cancel_direct − len(explicit_excluded_ids)`
   - `after_outlier == after_explicit_excluded − len(outlier_ids)`
   - `sum(outlier_reason counts) == after_explicit_excluded − after_outlier`
   - `cancel_only + direct_only + cancel_and_direct + after_cancel_direct_count == raw`
10. **dtype 어서션**: `assert cleaned["floor"].dtype == "Int16"` 외 Phase 1 컬럼 전체.
11. **cancel/direct overlap 분해 단위 테스트 (v6 추가)** — 위치: [`tests/test_data_preprocessing.py`](tests/test_data_preprocessing.py:8). 기존 테스트는 legacy `cancel_trade_count` / `direct_trade_count` 만 확인하므로, 변경 6 의 신규 4개 컬럼(`cancel_only_count`, `direct_only_count`, `cancel_and_direct_count`, `after_cancel_direct_count`) 과 귀속 규칙을 직접 검증하는 케이스를 추가한다.
    - raw fixture: 한 연도·지역에 4종 행을 모두 포함:
      - `cancel only` (`cdealType="O"`, `dealingGbn="중개"`)
      - `direct only` (`cdealType=" "`, `dealingGbn="직거래"`)
      - `cancel + direct` (`cdealType="O"`, `dealingGbn="직거래"`)
      - `neither` (`cdealType=" "`, `dealingGbn="중개"`)
    - 어서션:
      - 신규 4개 컬럼 값이 fixture 상의 실제 카운트와 정확히 일치.
      - `cancel_only + direct_only + cancel_and_direct + after_cancel_direct_count == total_trade_count` (raw 항등식).
      - `cancel_trade_count == cancel_only + cancel_and_direct`, `direct_trade_count == direct_only + cancel_and_direct` (legacy 하위호환).
      - funnel 귀속: `removed_cancel == cancel_only + cancel_and_direct`, `removed_direct == direct_only` (변경 6 잠금 규칙).
      - `preprocess_trade()` 실행 후 `apt_trade_{YYYY}.parquet` 의 행수 == `after_cancel_direct_count` (정합성 게이트와 동일한 불변식).

## 작업 순서

1. **변경 1**: `build_snapshot_outliers` 에 `return_verdicts` + row_id fallback. 기존 2-tuple 호출부 회귀 테스트.
2. **방어 조치 (변경 9 in-PR 부분)**: `pipelines/market_snapshot/io.py`, `pipelines/aggregation_pipeline.py:182`, `analysis/common.py:191` 글롭을 자릿수 정규 패턴으로 좁힘.
3. **변경 2 + 3**: `pipelines/cleaned_trade_pipeline.py` 작성 — 분류 (A-3 와 동일한 floor 캐스팅), partition completeness, dtype 최적화, per-file atomic write, `_SUCCESS` manifest.
4. **변경 6**: 요약 산출물 (long format) + `trade_filter_yearly_summary.parquet` join + 정합성 게이트.
5. **변경 5**: `scripts/build_cleaned_trade.py` 작성 + `run_preprocessing()` 직후 hook + `--strict-cleaned-trade` / `CI` env 분기.
6. **변경 8**: `.gitignore` 갱신.
7. **테스트**: 위 11개 회귀 테스트 (idempotency 는 content+schema 비교, overlap 분해는 `tests/test_data_preprocessing.py` 에 추가).
8. **변경 7**: 대시보드 page 15 — `📍 시장 한눈에 보기` 그룹 page 14 옆 + `st.info` empty state.
9. **검증 실행**: 한 번 돌려 행수 / 파일 크기 / dtype / 진단 요약 / 페이지 렌더 + manifest 확인.
10. **PR 분리**: cutover (변경 9 의 prefix 인자 전환 부분) 는 다음 PR.

## 명시적 비-목표

- 새로운 이상치 판정 로직 추가 ✗ — A-3 결과 그대로 채택.
- A-3 단계 자체의 파라미터 튜닝 ✗ — 별도 PR.
- 전월세(`apt_rent`) 정제 ✗ — 본 PR 은 매매만.
- 기존 분석 코드의 입력 전환 (cutover) ✗ — 본 PR 은 데이터셋만 생성. cutover 는 follow-up PR.
- canonical schema 의 `category` 변환 ✗ — Phase 2 로 분리.

## 검토 포인트

v4 에서 모두 잠겼다. 기록만 남긴다.

1. ~~explicit_excluded 처리~~ → **확정**: canonical cleaned 에서 제외, summary 와 로그에서만 추적. 디버깅 필요 시 별도 audit 샘플 산출물 추가 (follow-up).
2. ~~funnel 페이지 시각화~~ → **확정**: 상단 funnel/waterfall (선택 연도) + 중단 stacked area (연도 추세) + 하단 outlier reason 분해. sankey 미채택.
3. ~~variant 산출물 정책~~ → **확정**: `data/processed/variants/` 하위 + gitignore.
4. ~~dtype Phase 2 (category) 시점~~ → **확정**: cutover PR 에도 넣지 않고, cutover 안정화 이후 별도 PR.
5. ~~`run_full_pipeline.py` 실패 정책~~ → **확정**: 로컬 기본 warning, `--strict-cleaned-trade` 또는 `CI=true` 시 fatal.

---

## v6 변경 요약 (review 응답)

### 1. Revised plan
위 본문이 v6 revised plan.

### 2. Key changes made (v5 → v6)

**리뷰 Findings 직접 반영**

- **MEDIUM `[변경 2/4]` step 6 문구 정정**: partition assertion 실패 시 "기존 `_SUCCESS` 는 이미 삭제된 상태" 라는 v4 잔존 문구 제거. v5 의 "`_SUCCESS` 삭제는 첫 `replace()` 직전" 정책과 일치하게, preflight/assertion 실패 시점에는 canonical 파일·`_SUCCESS` 모두 무손상이며 직전 validated dataset 을 계속 신뢰할 수 있음을 명시.
- **MEDIUM `[변경 4/7]` manifest `files[].path` 기준 고정**: `data/` 기준 상대경로 + POSIX 구분자로 잠금. 예: `processed/cleaned_apt_trade_2010.parquet`, `preprocessed_plus/cleaned_trade_yearly_summary.parquet`. summary 2종도 manifest 에 등재. `_SUCCESS` 자체는 자기 검증 불가로 미등재. `get_cleaned_trade_manifest(data_root=...)` 시그니처가 해석 기준점.
- **MEDIUM `[변경 6]` overlap 분해 단위 테스트 추가**: 테스트 #11 신설. `tests/test_data_preprocessing.py` 에 4종 fixture (cancel only / direct only / cancel+direct / neither) + 신규 4개 컬럼 값 + legacy 하위호환 + funnel 귀속 규칙 + 정합성 게이트 불변식까지 검증.
- **LOW `[변경 7]` empty-state 중복 제거**: 뒤쪽 generic 버전 삭제. manifest-gated v5 버전만 유지.

### 3. Remaining open questions

없음. v6 는 구현 spec.

---

## v5 변경 요약 (review 응답)

### 1. Revised plan
위 본문이 v5 revised plan (v6 로 갱신됨).

### 2. Key changes made (v4 → v5)

**리뷰 Findings 직접 반영**

- **HIGH `[변경 6/7]` cancel/direct overlap 귀속 + page 15 데이터 소스 잠금**: preprocessing 의 실제 처리 순서(cancel→direct, [`pipelines/data_preprocessing.py:328`](pipelines/data_preprocessing.py:328) → [`:335`](pipelines/data_preprocessing.py:335))를 그대로 반영해 `removed_cancel = cancel_only + cancel_and_direct`, `removed_direct = direct_only` 로 잠금. page 15 는 `trade_filter_yearly_summary` + `cleaned_trade_yearly_summary` + `cleaned_trade_outlier_reason_summary` 3-way join 으로 명시. denormalized summary 별도 추가 안 함 (manifest validity 단일 진실 소스 유지).
- **MEDIUM `[변경 4]` `_SUCCESS` 삭제 시점 지연**: "빌드 시작 직후" → "모든 preflight + assertion 통과 + 모든 `*.tmp` 작성 완료 직후, 첫 `replace()` 직전". preflight 단계 실패 시 직전 validated dataset 의 availability 보존. correctness 는 그대로 유지.
- **MEDIUM `[변경 4/7]` 로더 분리**: `get_cleaned_trade_manifest()` (validity 만, 청크 미로드) 와 `load_cleaned_trade()` (본문) 를 분리. page 15 는 manifest gate 만 사용, summary parquet 만 로드 → 책임 분리 + 페이지 무게 감소.
- **LOW `[테스트]` 8/8b 문구 정정**: v4 의 size+mtime+sha256 + `_SUCCESS` 존재 기반 validity 와 일치하도록 단일 항목으로 정리. 진실 소스가 timestamp 가 아니라 manifest 검증임을 명시.

### 3. Remaining open questions

없음. v5 는 구현 spec 으로 진입 가능.
