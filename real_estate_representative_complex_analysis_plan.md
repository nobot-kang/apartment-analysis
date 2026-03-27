# 59형·84형 대표단지 평당가 분석 레이어 추가 계획

## Summary
- 기존 `price_per_m2`/`price_per_py` 전처리는 유지하고, 새로 `59형/84형 대표단지 평당가` 전용 집계·분석 계층을 병렬 추가한다.
- 대표단지 universe는 `이중 집합`으로 고정한다.
  - 기본 대표단지: 59형 또는 84형을 가진 단지
  - pair 단지: 59형과 84형을 모두 가진 단지
- 대표면적 band는 raw `area` 기준으로 고정한다.
  - `59형`: `58.0 <= area < 60.0`
  - `84형`: `83.0 <= area < 85.0`
- 거래가 없는 월은 `이전 관측 시세를 forward-fill`하여 시세가 유지된다고 가정한다.
  - `backfill`은 하지 않는다.
  - fill 상한은 `12개월`로 고정한다.
  - 12개월을 초과한 공백은 다시 결측으로 둔다.
- 같은 단지 내 `59형 vs 84형` gap은 `band별 월별 대표 평당가를 먼저 ffill`한 뒤, 그 시계열에 `trailing 3개월 rolling median`을 적용해서 계산한다.
- 지역 대표가격도 raw 거래 평균이 아니라 `단지별 대표값 -> 지역 집계`의 2단계 구조로 계산한다.

## Key Changes
### Data layer
- `AggregationPipeline`에 아래 parquet를 추가한다.
  - `representative_complex_universe.parquet`
    - 키: `aptSeq`
    - 컬럼: 단지 기본정보, `has_trade_59`, `has_trade_84`, `has_rent_59`, `has_rent_84`, `has_59_any`, `has_84_any`, `is_pair_complex`, band별 관측월 수, 최초/최종 관측월
  - `representative_trade_band_monthly.parquet`
    - 키: `aptSeq, ym, area_band`
    - full monthly panel로 저장
    - 컬럼: `trade_count_obs`, `price_per_py_obs_median`, `price_per_py_obs_mean`, `price_per_py_obs_trimmed`, `price_per_py_filled`, `months_since_trade_obs`, `is_trade_imputed`, `fill_active`, 단지/지역/건축물대장 특성
  - `representative_rent_band_monthly.parquet`
    - 키: `aptSeq, ym, area_band, rentType`
    - full monthly panel로 저장
    - 컬럼: `rent_count_obs`, `deposit_per_py_obs_median`, `monthly_rent_per_py_obs_median`, `deposit_per_py_filled`, `monthly_rent_per_py_filled`, `months_since_rent_obs`, `is_rent_imputed`, `fill_active`
  - `representative_pair_gap_monthly.parquet`
    - 키: `aptSeq, ym`
    - 컬럼: `sale_py_59_roll3`, `sale_py_84_roll3`, `sale_gap_abs`, `sale_gap_ratio`, `sale_gap_log`, `sale_59_fill_age`, `sale_84_fill_age`, `sale_any_imputed`
    - 전세/월세도 같은 구조로 `jeonse_*`, `wolse_*` 컬럼 추가
  - `representative_region_monthly.parquet`
    - 키: `region_level, region_code, ym, area_band, market_type`
    - `region_level`: `sigungu` / `bjdong`
    - `market_type`: `sale` / `jeonse` / `wolse`
    - 컬럼: `complex_count_active`, `complex_count_observed`, `complex_eq_median_py`, `complex_eq_mean_py`, `complex_eq_trimmed_py`, `tx_weighted_mean_py`, `p10`, `p25`, `p75`, `p90`, `iqr`, `transaction_count_obs`
    - 지역 집계는 `filled` 값을 기본 입력으로 쓰고, 실제 관측이 있었던 단지 수를 별도 컬럼으로 저장
  - `representative_forecast_targets.parquet`
    - 키: `aptSeq, ym`
    - 컬럼: band별 lag, gap lag, 거시지표, 단지특성, 미래 1/3/12개월 타깃
- 월별 band panel 생성 규칙은 다음으로 고정한다.
  - 먼저 `aptSeq + ym + area_band`에서 raw `price_per_py` median을 계산
  - 그 결과를 단지별 전체 월 캘린더로 reindex
  - 마지막 실제 관측값을 `limit=12`로 forward-fill
  - `months_since_*`와 `is_*_imputed`를 함께 저장
  - pre-first-observation 구간은 결측 유지

### Analysis layer
- 새 대표단지 분석 모듈을 추가한다.
- pair gap 계산 규칙은 다음으로 고정한다.
  - `price_per_py_filled` 기준으로 band별 월 series 생성
  - 그 위에 `trailing 3개월 rolling median(min_periods=1)` 적용
  - 두 band 모두 non-null이면 gap 계산
  - 한 band가 fill 만료로 null이면 해당 월 gap은 null
- 지역 분포 분석은 항상 `representative_region_monthly.parquet`를 사용한다.
- 예측 타깃은 다음으로 고정한다.
  - `future_sale_py_59_return_{1,3,12}m`
  - `future_sale_py_84_return_{1,3,12}m`
  - `future_sale_gap_ratio_change_{1,3,12}m`
- feature는 기존 `complex_master` 특성을 재사용한다.
  - `parking_per_household`, `floor_area_ratio`, `building_coverage_ratio`, `avg_land_area_per_household`, `household_count`, `complex_age`, `redevelopment_option_score`

### Dashboard surface
- 대표단지 분석 그룹 4개를 추가하고 각 그룹 5개 섹션, 총 20개 분석으로 구성한다.
- 모든 페이지는 lazy-loading 유지, precomputed parquet만 읽는다.
- 기본 필터는 `서울/경기/수도권`, `시군구`, `법정동`, `59/84/pair`, `최근 6/12/24개월`, `매매/전세/월세`로 통일한다.
- 지역 시계열 차트에는 `active 단지 수`와 `observed 단지 수`를 함께 표시해 fill 의존도를 보이도록 한다.

## 20 Analyses
### Level 1 · 대표단지 기초 현황
1. 대표단지 coverage: 시군구/법정동별 59형, 84형, pair 단지 수와 비중.
2. 지역별 평당가 분포 타임라인: 시군구/법정동별 `complex_eq_median_py`와 분위수 밴드.
3. 59형 vs 84형 지역 추이 비교: 같은 지역에서 band별 평당가 수준과 spread.
4. 특정 시점 분포도: 시군구/법정동별 box/violin 분포.
5. pair 단지 gap 히스토리: 같은 단지 내 `sale_gap_ratio` 3개월 rolling 추이.

### Level 2 · 가격 구조와 단면 비교
6. 지역 spread decomposition: 지역별 `84/59` 평당가 비율과 절대격차.
7. 전세 평당가 분포 및 spread: `deposit_per_py` 기준 59/84 비교.
8. 월세 평당가 분포 및 spread: `monthly_rent_per_py` 기준 59/84 비교.
9. 전세가율 band 비교: 59형/84형별 전세가율과 같은 단지 내 gap.
10. liquidity/coverage 진단: 거래발생률, fill 의존도, pair 지속성.

### Level 3 · 변화 원인과 동학
11. pair gap rolling coefficient: 단지특성이 gap에 주는 영향의 시변 계수.
12. 단지 고정효과 패널: 동일 단지 내부 59/84 평당가 변화와 거시변수의 결합.
13. 거시 국면별 spread 반응: 금리/M2/환율 국면에서 59형과 84형 민감도 차이.
14. 지역 확산 분석: 법정동별 대표 spread 변화의 인접지역 전이.
15. mean reversion / persistence: gap 확대 후 축소 속도와 지속기간.

### Level 4 · 예측과 시뮬레이션
16. 59형 평당가 예측: 1/3/12개월 앞 `sale_py_59`.
17. 84형 평당가 예측: 1/3/12개월 앞 `sale_py_84`.
18. pair gap 비율 예측: 1/3/12개월 앞 `sale_gap_ratio` 변화.
19. 지역 screening: 향후 `59 강세`, `84 강세`, `spread 확대`, `spread 축소` 지역/단지 랭킹.
20. 시나리오 시뮬레이터: 금리/M2/환율 shock 시 59형, 84형, gap 경로 비교.

## Test Plan
- 데이터 정의
  - `price_per_py`를 재계산하지 않고 기존 전처리 값과 일치하는지 샘플 검증
  - band 매핑이 raw `area` 기준 `[58,60)`, `[83,85)`로 동작하는지 경계값 검증
  - `representative_complex_universe`에서 `is_pair_complex == has_59_any & has_84_any` 검증
- fill 로직
  - 관측월 직후 1~12개월은 이전 시세가 유지되는지 검증
  - 13개월째는 fill 이 끊기고 null로 복귀하는지 검증
  - 최초 관측 이전 월은 backfill 되지 않는지 검증
  - `months_since_*`, `is_*_imputed`, `fill_active`가 실제 fill 상태와 일치하는지 검증
- pair gap
  - 한 band가 관측이 없어도 12개월 이내면 이전 시세를 사용해 gap이 계산되는지 검증
  - fill 만료 후에는 gap이 null로 전환되는지 검증
  - 3개월 rolling median이 `filled` 시계열 위에서 계산되는지 샘플 검증
- 집계/병합
  - `representative_trade_band_monthly`: `aptSeq + ym + area_band` 중복 0
  - `representative_rent_band_monthly`: `aptSeq + ym + area_band + rentType` 중복 0
  - `representative_pair_gap_monthly`: `aptSeq + ym` 중복 0
  - `representative_region_monthly`에서 `complex_count_active >= complex_count_observed` 검증
- 대시보드
  - 지역 분포 페이지, pair gap 페이지, 전세/월세 spread 페이지, 예측 페이지 smoke test
  - 새 페이지는 precomputed parquet만 읽고 5초 안쪽 렌더를 목표로 측정
  - headless Streamlit import/render smoke 실행

## Assumptions
- 기존 지역 평균 로직은 교체하지 않고, 대표단지 평당가 분석 계층을 별도로 추가한다.
- `forward-fill`은 사용자가 요청한 정책으로 반영하되, 장기 공백 왜곡을 막기 위해 `12개월 상한`을 둔다.
- pair gap의 기본 기준은 `매매 평당가`이며 전세/월세는 같은 구조를 병렬 적용한다.
- 지역 대표가격의 기본값은 `complex-equal-weight median`이고, 거래가중 평균은 보조 지표로만 제공한다.
- 새 UI는 기존 lazy-loading 구조를 유지한 채 별도 대표단지 분석 그룹으로 추가한다.
