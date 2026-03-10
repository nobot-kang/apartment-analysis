# 국토교통부 아파트매매 실거래 상세 자료 API 컬럼 정의

| 영문 컬럼명 | 한글 의미 | 비고 |
|---|---|---|
| sggCd | 법정동시군구코드 | |
| umdCd | 법정동읍면동코드 | |
| landCd | 법정동지번코드 | |
| bonbun | 법정동본번코드 | |
| bubun | 법정동부번코드 | |
| roadNm | 도로명 | |
| roadNmSggCd | 도로명시군구코드 | |
| roadNmCd | 도로명코드 | |
| roadNmSeq | 도로명일련번호코드 | |
| roadNmbCd | 도로명지상지하코드 | |
| roadNmBonbun | 도로명건물본번호코드 | |
| roadNmBubun | 도로명건물부번호코드 | |
| umdNm | 법정동 | |
| aptNm | 단지명 | |
| jibun | 지번 | |
| excluUseAr | 전용면적 | |
| dealYear | 계약년도 | |
| dealMonth | 계약월 | |
| dealDay | 계약일 | |
| dealAmount | 거래금액 | (만원) |
| floor | 층 | |
| buildYear | 건축년도 | |
| aptSeq | 단지 일련번호 | |
| cdealType | 해제여부 | |
| cdealDay | 해제사유발생일 | |
| dealingGbn | 거래유형 | (중개및직거래여부) |
| estateAgentSggNm | 중개사소재지 | (시군구단위) |
| rgstDate | 등기일자 | |
| aptDong | 아파트 동명 | |
| slerGbn | 거래주체정보_매도자 | (개인/법인/공공기관/기타) |
| buyerGbn | 거래주체정보_매수자 | (개인/법인/공공기관/기타) |
| landLeaseholdGbn | 토지임대부 아파트 여부 | |

---

> **참고**: 위 컬럼은 국토교통부 API에서 제공하는 원본 데이터(`data/raw/molit/apt_trade/...`)의 정의입니다. 
> 분석을 위해 전처리된 데이터(`data/processed/apt_trade.parquet`)의 컬럼 정의는 `data/docs/processed_data_schema.md`를 참고하세요.
