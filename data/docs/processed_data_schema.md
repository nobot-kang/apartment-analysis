# 가공 데이터 스키마 (Processed Data Schema)

`data_processed` 디렉토리에 저장된 파케이(Parquet) 파일들의 스키마 및 컬럼 정보입니다.

## 1. 아파트 매매 데이터 (`apt_trade.parquet`)

국토교통부 매매 실거래가 데이터를 전처리하여 저장한 파일입니다.

| 컬럼명 | 데이터 타입 | 설명 | 단위/비고 |
|---|---|---|---|
| date | datetime64[ns] | 거래 일자 | YYYY-MM-DD |
| price | int64 | 거래 금액 | 만원 |
| price_per_m2 | float64 | ㎡당 가격 | 만원/m² (탐색·이상치 탐지용) |
| price_per_py | float64 | 평당 가격 | 만원/평 (1평=3.3058m²) |
| price_std84 | Int64 | 84㎡ 환산 가격 | 만원 (선형 비례 환산) |
| area | float64 | 전용면적 | m² (분석용 원본) |
| floor | float64 | 층수 | 층 |
| construction_year | int64 | 건축년도 | 년 |
| age | float64 | 건물 경과연수 | 년 (거래시점 기준) |
| dong | object | 법정동명 (원본) | 예: 신림동 |
| apt_name | object | 아파트 단지명 (원본) | 예: 신림현대 |
| area_repr | Int64 | 대표 전용면적 (평형 그룹핑용) | m² (예: 59, 84) |
| dong_repr | object | 대표 법정동명 (분석용) | 예: 신림동(11620) |
| apt_name_repr | object | 대표 단지명 (분석용) | 예: 신림현대(신림동) |
| aptSeq | object | 단지 고유 식별자 | 국토부 단지코드 |

> **참고**: `price_std84`는 단순 선형 비례(`price × 84/area`) 환산이다. 면적이 커질수록 ㎡당 단가가 떨어지는 규모효과(β<1)를 반영하려면, 단지별/법정동별 면적 탄력성 β를 추정하여 `price × (84/area)^β` 로 보정해야 한다. `area_repr`는 평형 그룹핑 표시용이며, 분석에서는 `area`를 사용할 것.

## 2. 아파트 전월세 데이터 (`apt_rent.parquet`)

국토교통부 전월세 실거래가 데이터를 전처리하여 저장한 파일입니다.

| 컬럼명 | 데이터 타입 | 설명 | 단위/비고 |
|---|---|---|---|
| date | datetime64[ns] | 거래 일자 | YYYY-MM-DD |
| deposit | int64 | 보증금 | 만원 |
| deposit_per_m2 | float64 | 보증금 ㎡당 가격 | 만원/m² |
| deposit_per_py | float64 | 보증금 평당 가격 | 만원/평 |
| deposit_std84 | Int64 | 보증금 84㎡ 환산 | 만원 (선형 비례 환산) |
| monthly_rent | int64 | 월세 | 만원 |
| monthly_rent_per_m2 | float64 | 월세 ㎡당 가격 | 만원/m² |
| monthly_rent_per_py | float64 | 월세 평당 가격 | 만원/평 |
| area | float64 | 전용면적 | m² (분석용 원본) |
| floor | float64 | 층수 | 층 |
| construction_year | int64 | 건축년도 | 년 |
| age | float64 | 건물 경과연수 | 년 (거래시점 기준) |
| dong | object | 법정동명 | |
| apt_name | object | 아파트 단지명 | |
| area_repr | Int64 | 대표 전용면적 (평형 그룹핑용) | |
| dong_repr | object | 대표 법정동명 | |
| apt_name_repr | object | 대표 단지명 | |
| aptSeq | object | 단지 고유 식별자 | |
| contract_type | object | 계약 구분 | 신규, 갱신 등 |
| contract_term | object | 계약 기간 | |
| use_rr_right | object | 갱신요구권 사용 여부 | |

## 3. 아파트 정보 조회 테이블 (`apartment_info.parquet`)

건축물대장(표제부) 데이터를 요약하여 아파트 단지의 특성을 저장한 lookup table입니다. `aptSeq`를 키로 하여 매매/전월세 데이터와 결합할 수 있습니다.

| 컬럼명 | 데이터 타입 | 설명 | 단위/비고 |
|---|---|---|---|
| aptSeq | object | 단지 고유 식별자 | 키 컬럼 |
| apt_name_ledger | object | 건축물대장상 단지명 | |
| completion_date | datetime64[ns] | 사용승인일 (준공일) | |
| land_area | float64 | 대지면적 | m² |
| floor_area_ratio_total_area | float64 | 용적률산정연면적 | m² |
| total_area | float64 | 연면적 | m² |
| floor_area_ratio | float64 | 용적률 | % |
| building_coverage_ratio | float64 | 건폐율 | % |
| household_count | float64 | 세대수 | 세대 |
| ground_floor_count | float64 | 지상층수 | 층 |
| underground_floor_count | float64 | 지하층수 | 층 |
| address | object | 대지위치 (주소) | |
| sigungu_code | object | 시군구코드 | |
| bjdong_code | object | 법정동코드 | |

## 4. 월별 통합 데이터 (`merged_monthly.parquet`)

아파트 거래 데이터의 월별 집계와 거시경제 지표를 통합한 데이터입니다.

| 컬럼명 | 데이터 타입 | 설명                 | 단위/비고 |
|---|---|--------------------|---|
| date | datetime64[ns] | 기준 월               | YYYY-MM-01 |
| price_mean | float64 | 아파트 평균 거래가         | 만원 |
| price_median | float64 | 아파트 중위 거래가         | 만원 |
| volume | int64 | 거래량                | 건 |
| area_mean | float64 | 평균 전용면적            | m² |
| base_rate | float64 | 한국 기준금리            | % |
| cpi_kr | float64 | 한국 소비자물가지수         | 2020=100 기준 |
| mortgage_rate | float64 | 주택담보대출금리 (신규취급액)   | % |
| m2_avg_season | float64 | M2 광의통화 (말잔, 계절조정) | 십억원 |
| cpi_us | float64 | 미국 소비자물가지수         | 1982-1984=100 기준 |
| base_rate_us | float64 | 미국 기준금리            | % |
| base_rate_eu | float64 | 유로존 기준금리           | % |
| base_rate_jp | float64 | 일본 기준금리            | % |
| gold_usd | float64 | 국제 금 시세            | USD/oz |
| usdkrw | float64 | 원/달러 환율            | 원 |

## 5. 법정동별 가격 추이 (`dong_price_trends.parquet`, `dong_price_trends_85.parquet`)

법정동 단위의 아파트 평당 가격 추이 데이터입니다. `_85` 파일은 한국 아파트 대표 전용면적인 59m² 과 84m² 이하(국민평형) 기준으로 환산 보정된 데이터입니다.

| 컬럼명 | 데이터 타입 | 설명 | 단위/비고 |
|---|---|---|---|
| month_date | datetime64[ns] | 기준 월 | YYYY-MM-01 |
| dong_repr | object | 법정동명 | |
| price_krw | float64 | 평당 가격 (원화) | 만원/평 (3.3m²) |
| usdkrw | float64 | 원/달러 환율 | 원 |
| gold_usd | float64 | 국제 금 시세 | USD/oz |
| price_usd | float64 | 평당 가격 (달러 환산) | USD/평 |
| price_gold_oz | float64 | 평당 가격 (금 환산) | oz/평 (금 몇 온스로 1평 구매 가능한지) |
| price_m2 | float64 | 평당 가격 대비 M2 비율 | 만원 / 십억원 (단순 비율) |

## 6. 단지별 대지 가격 추이 (`complex_land_price_trends.parquet`)단지별 대지면적을 고려하여 계산한 **대지 평당가** 추이 데이터입니다. 건물이 아닌 땅(대지)의 가치 변화를 분석하기 위해 사용합니다.

| 컬럼명 | 데이터 타입 | 설명 | 단위/비고 |
|---|---|---|---|
| month_date | datetime64[ns] | 기준 월 | YYYY-MM-01 |
| apt_name_repr | object | 대표 단지명 | 예: 신림현대(신림동) |
| dong_repr | object | 법정동명 | |
| land_share_repr | int64 | 평균 대지지분 | 평 (올림 처리) |
| price_krw | float64 | 대지 평당 가격 (원화) | 만원/평 (3.3m²) |
| usdkrw | float64 | 원/달러 환율 | 원 |
| gold_usd | float64 | 국제 금 시세 | USD/oz |
| price_usd | float64 | 대지 평당 가격 (달러 환산) | USD/평 |
| price_gold_oz | float64 | 대지 평당 가격 (금 환산) | oz/평 |
| price_m2 | float64 | 대지 평당 가격 대비 M2 비율 | 만원 / 십억원 |
