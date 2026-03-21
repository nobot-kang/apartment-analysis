# 국토교통부 아파트전월세 실거래 상세 자료 API 컬럼 정의

| 영문 컬럼명 | 한글 의미 | 비고 |
|---|---|---|
| sggCd | 시군구코드 | |
| umdNm | 법정동 | |
| aptNm | 단지명 | |
| jibun | 지번 | |
| excluUseAr | 전용면적 | |
| dealYear | 계약년도 | |
| dealMonth | 계약월 | |
| dealDay | 계약일 | |
| deposit | 보증금 | (만원) |
| monthlyRent | 월세 | (만원) |
| floor | 층 | |
| buildYear | 건축년도 | |
| aptSeq | 단지 일련번호 | |
| contractTerm | 계약기간 | |
| contractType | 계약구분 | (신규/갱신) |
| useRRRight | 갱신요구권 사용여부 | |
| preDeposit | 종전보증금 | |
| preMonthlyRent | 종전월세 | |
| roadnm | 도로명 | |
| roadnmcd | 도로명코드 | |
| roadnmsggcd | 도로명시군구코드 | |
| roadnmseq | 도로명일련번호코드 | |
| roadnmbcd | 도로명지상지하코드 | |
| roadnmbonbun | 도로명건물본번호코드 | |
| roadnmbubun | 도로명건물부번호코드 | |

---

> **참고**: 위 컬럼은 국토교통부 API에서 제공하는 원본 데이터(`data/raw/molit/apt_rent/...`)의 정의입니다. 
> 분석을 위해 전처리된 데이터의 컬럼 정의는 `data/docs/processed_data_schema.md`를 참고하세요.
