# A-3 이상치 탐지 로직 업데이트 계획

대상 파일
- `pipelines/market_snapshot_pipeline.py` (`build_snapshot_outliers`, `_compute_monthly_band_frame`, `_annotate_trend_confirmation`)
- 결과 소비처: [dashboard/pages/page_00_market_snapshot_diagnostics.py:303](dashboard/pages/page_00_market_snapshot_diagnostics.py#L303) (`_render_a3`)

---

## 1. 현재 로직 요약

| 단계 | 내용 |
|------|------|
| 그룹 단위 | `aptSeq × area_repr × month` |
| 월 대표값 | `month_price_m2 = median(price_per_m2)` + trade_count / std / MAD |
| 기준 시세 (`ref_price`) | 직전 6개월 lag median의 trailing MA (min 3개월) |
| 밴드 폭 | `max(rolling_std × 2, ref_price × 25%)` |
| Stale 처리 | `ref_gap_months > 6` 이면 ref/std를 NaN 처리 |
| 1차 판정 | 월 대표값이 band 밖이면 candidate |
| 추세 전환 복원 | 이후 6개월 내 2개월 + 3건 이상 같은 방향 유지 시 outlier 해제, 새 레벨로 태깅 |
| 추세 전환 월 재점검 | 월 중앙값 기준 `MAD × 2.5` vs `±8%` 중 큰 값으로 row 단위 재판정 |
| 선제 제외 | 1층 거래 |

**핵심 한계**
1. **희소 그룹 무력화.** `aptSeq × area_repr` 기준 직전 3개월 관측이 없으면 `ref_price`가 NaN → 모든 거래가 통과. 거래가 듬성듬성한 구형·비대표 평형에서 오탐/누락이 동시에 발생.
2. **Trailing-only 레퍼런스.** 직전 월만 사용하므로 시세 변동 시점에서 "고점 직후 하락" 같은 패턴을 이상치로 오판. Forward-looking 정보는 trend confirmation 단계에서만 활용.
3. **단일 스케일 밴드.** 그룹 내부 분산에만 의존. 인접 단지·행정동의 정상 변동폭을 반영하지 않아, 한 건만 있는 월은 그 한 건이 곧 "시세"가 되어 비교 불가.
4. **시세 곡선의 부재.** 월 중앙값의 트레일링 평균일 뿐, 매끄럽게 변동하는 "시세 경로"가 명시적으로 존재하지 않음. 사용자가 요구한 "시간에 따라 자연스레 변동하는 시세"와 격차.
5. **추세 전환 판정이 후행.** 최소 2개월 + 3건 거래가 쌓여야 복원 → 초기 월에 정상 상승/하락분이 이상치로 잡힐 수 있음.

---

## 2. 업데이트 목표

1. 시간에 따라 매끄럽게 변동하는 **시세 경로 (smoothed market path)** 를 그룹별로 명시적으로 구축한다.
2. 희소 그룹은 **상위 코호트 (행정동 × area_bucket, 시군구 × area_bucket)** 시세 경로로 **shrinkage** 하여 안정적인 참조값을 확보한다.
3. 시세 경로로부터 **유의미하게 큰 편차 + 고립된 스파이크**만 이상치로 판정한다 (자연스런 추세 변동은 유지).
4. 진단 컬럼을 풍부하게 남겨 대시보드에서 사유를 추적 가능하게 한다.

---

## 3. 업데이트 설계

### 3.1 데이터 준비 (변경 없음 + 추가)
- 기존 전처리 유지: `1층 제외`, `price_per_m2`, `area_repr`.
- **추가 계층 키 생성**: `sggCd`, `dong_code`, `area_bucket`. 상위 코호트 시세 계산에 사용.

### 3.2 다층 시세 경로 구축

| 계층 | 집계 단위 | 목적 |
|------|-----------|------|
| L0 | `aptSeq × area_repr × month` | 그룹 고유 월 중앙값 (현재와 동일) |
| L1 | `sggCd × area_bucket × month` | 광역 시세 fallback |
| L2 | `dong_code × area_bucket × month` | 동 단위 fallback (중간 해상도) |

각 계층에서 **log(price_per_m2)** 에 대해 **centered rolling median (window=7, min=3)** 또는 **LOWESS (frac≈0.15)** 를 적용해 `path_{level}` 을 생성. 로그 공간에서 곡선을 그리면 상대 편차(%)로 해석이 용이.

> 선택지: LOWESS (scikit-learn/statsmodels) vs centered rolling median. 1차 구현은 의존성을 최소화하기 위해 **centered rolling median + EWM 보강**으로 시작하고, advanced 옵션으로 LOWESS를 교체 가능하도록 함수화.

### 3.3 Shrinkage 기반 그룹 시세

그룹 월별 관측량 `n_g`와 직전 6개월 누적 거래수 `N_g` 를 이용해 가중치 계산:

```
w_group = N_g / (N_g + k)        # k=6 기본값
ref(t)  = w_group * path_L0(t) + (1 - w_group) * path_L2_or_L1(t)
```

- 관측이 충분한 그룹: 자체 path 우선
- 희소 그룹: 상위 코호트 path에 자동으로 수렴
- L2가 비어있으면 L1으로 폴백
- 모든 계산은 log-space 후 지수 복원

### 3.4 동적 밴드

- 그룹 MAD (`robust sigma`) 가 충분히 크면 그것을 사용, 부족하면 동일 코호트 MAD 로 대체.
- 밴드 폭:
  ```
  sigma_eff  = max(robust_sigma_group, robust_sigma_cohort)
  band_abs   = max(z * sigma_eff, ref * floor_pct)
  ```
  - `z = 3.0` (기본, 현재 2.0 → 3.0 로 완화해 과도한 false positive 축소)
  - `floor_pct = 0.20` (최소 ±20%)
- 희소 그룹에 대해서는 `floor_pct`를 `+0.05` 만큼 확대 (예: n ≤ 3 이면 0.25).

### 3.5 이상치 판정 2단계

1. **거친 필터 (global sanity check)**
   - `|log(price) - log(path_L1)| > log(2.0)` (≒ 2배 이상 차이) → 즉시 제거 후보.
   - 명백한 입력 오류 / 단위 이상 제거.

2. **정밀 필터 (row-level)**
   - `|price - ref| > band_abs` 이고, 동시에
   - 그룹 내 **고립 스파이크**인지 확인:
     - 동일 방향 인접 2개월 내 지지 거래가 `support_ratio < 0.3` 이면 고립 → 이상치 확정
     - 지지 거래가 많으면 **추세 전환**으로 해제 (기존 로직 계승 및 리팩터)

### 3.6 추세 전환 판정 개선
- 기존 forward-only 판정에 **backward confirmation** 추가: 이상 월 직전 1~2개월에서 같은 방향 근접 관측이 있으면 연속 추세로 인정.
- 추세로 인정된 월 내부의 개별 row 재점검은 유지하되, 임계값을 그룹별 동적 밴드와 동일 기준으로 통합.

### 3.7 출력 스키마 추가

`snapshot_outliers.parquet` 에 다음 진단 필드를 추가:

| 컬럼 | 의미 |
|------|------|
| `path_group_m2` | 그룹 자체 smoothed path |
| `path_cohort_m2` | 코호트 (L2/L1) smoothed path |
| `ref_price_shrunk` | Shrinkage 적용 최종 ref |
| `shrink_weight` | `w_group` |
| `sigma_eff` | 밴드 계산에 쓰인 효과 표준편차 |
| `band_width_pct` | 기존 유지 |
| `price_deviation_pct` | 기존 유지 (단, ref_price_shrunk 기준) |
| `outlier_reason` | `global_sanity` / `isolated_spike` / `trend_row_robust` |
| `cohort_level` | `L0` / `L2` / `L1` — 어떤 레벨로 폴백되었는지 |
| `group_trade_count_6m` | `N_g`, 희소성 판정용 |

기존 컬럼 (`ref_price`, `band_width_abs`, `outlier_direction`, `is_outlier`) 은 하위 호환 위해 유지.

---

## 4. 구현 단계

1. **설정 상수 리팩터**: `OUTLIER_*`, `BOLLINGER_*`, `TREND_*`, 신규 `SHRINK_K`, `BAND_Z`, `FLOOR_PCT`, `SANITY_LOG_RATIO` 를 한 곳에 모으고 dict 로 노출 (테스트 주입용).
2. **시세 경로 함수**: `_build_path_frame(df, keys, window)` — 로그 공간 centered rolling median. 세 개 계층 모두 동일 함수로 생성.
3. **Shrinkage 병합 함수**: `_merge_paths_with_shrinkage(group_path, cohort_path, n_counts)` → `ref_price_shrunk`, `shrink_weight`, `cohort_level`.
4. **밴드 계산**: `_compute_dynamic_band(monthly, cohort_sigma)` — sigma_eff, band_abs 산출.
5. **판정 파이프라인 재작성**:
   - `global_sanity_mask` → hard drop
   - `isolated_spike_mask` → 일반 이상치
   - 기존 trend confirmation 로직을 `_annotate_trend_confirmation` 에서 분리해 재사용.
6. **대시보드 페이지 보강** (`page_00_market_snapshot_diagnostics.py`)
   - KPI에 `희소 그룹 비율`, `global_sanity 제거 건수` 표시
   - 이상치 타입 필터 (`outlier_reason` selectbox)
   - 시세 경로 vs 거래 산점도 (단지 드릴다운용 선택 UI)
7. **테스트 추가** (`tests/test_market_snapshot_pipeline.py` 에 이어서)
   - 희소 그룹이 코호트 path로 폴백하는지
   - global_sanity 2배 편차 제거 동작
   - 추세 전환이 복원되는지 (backward/forward)
   - 단일 거래 월이 자기 자신으로 band를 만들지 않는지

---

## 5. 롤아웃 & 검증

1. 기존 `snapshot_outliers.parquet` 을 `snapshot_outliers_legacy.parquet` 으로 백업.
2. 신규 로직으로 재생성 후 `notebook/02_a3_bollinger_outlier_review.ipynb` 에서 legacy vs new 의
   - 이상치 건수 차이
   - 지역·평형별 분포 변화
   - 샘플 케이스 (상위 200건) 육안 비교
3. 회귀 확인 후 dashboard 반영, legacy 파일은 한 사이클 뒤 제거.

---

## 6. 파라미터 초기값 제안

| 파라미터 | 제안값 | 비고 |
|----------|--------|------|
| `PATH_WINDOW_MONTHS` | 7 (centered) | L0/L1/L2 공통 |
| `PATH_MIN_PERIODS` | 3 | centered window 내부 |
| `SHRINK_K` | 6 | 거래 6건이면 자체 path 가중치 50% |
| `BAND_Z` | 3.0 | 현행 2.0 → 완화 |
| `FLOOR_PCT_BASE` | 0.20 | 희소 그룹은 +0.05 |
| `SANITY_LOG_RATIO` | `ln(2.0)` | 2배 이상 편차는 즉시 제거 |
| `COHORT_L2` | `dong_code × area_bucket` | `area_bucket` 은 기존 AREA_BUCKETS 재사용 |
| `COHORT_L1` | `sggCd × area_bucket` | L2 비어있을 때만 |
