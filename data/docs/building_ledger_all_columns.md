# 건축물대장 표제부(아파트 단지 정보) 컬럼 정의

`bulding_ledger_all.parquet` 파일에 저장된 아파트 단지별 상세 정보(표제부) 컬럼 정의입니다.

| 영문 컬럼명 | 한글 의미 | 비고 |
|---|---|---|
| itgBldGrade | 지능형건축물등급 | |
| itgBldCert | 지능형건축물인증점수 | |
| crtnDay | 생성일자 | |
| naBjdongCd | 새주소법정동코드 | |
| naUgrndCd | 새주소지상지하코드 | |
| naMainBun | 새주소본번 | |
| naSubBun | 새주소부번 | |
| platArea | 대지면적(m²) | |
| archArea | 건축면적(m²) | |
| bcRat | 건폐율(%) | |
| totArea | 연면적(m²) | |
| vlRatEstmTotArea | 용적률산정연면적(m²) | |
| vlRat | 용적률(%) | |
| mainPurpsCd | 주용도코드 | |
| mainPurpsCdNm | 주용도코드명 | 공동주택 등 |
| etcPurps | 기타용도 | |
| hhldCnt | 세대수(세대) | |
| fmlyCnt | 가구수(가구) | |
| mainBldCnt | 주건축물수 | |
| atchBldCnt | 부속건축물수 | |
| atchBldArea | 부속건축물면적(m²) | |
| totPkngCnt | 총주차수 | |
| indrMechUtCnt | 옥내기계식대수(대) | |
| indrMechArea | 옥내기계식면적(m²) | |
| oudrMechUtCnt | 옥외기계식대수(대) | |
| oudrMechArea | 옥외기계식면적(m²) | |
| indrAutoUtCnt | 옥내자주식대수(대) | |
| indrAutoArea | 옥내자주식면적(m²) | |
| oudrAutoUtCnt | 옥외자주식대수(대) | |
| oudrAutoArea | 옥외자주식면적(m²) | |
| pmsDay | 허가일 | |
| stcnsDay | 착공일 | |
| useAprDay | 사용승인일 | 준공일자 |
| pmsnoYear | 허가번호년 | |
| pmsnoKikCd | 허가번호기관코드 | |
| pmsnoKikCdNm | 허가번호기관코드명 | |
| pmsnoGbCd | 허가번호구분코드 | |
| pmsnoGbCdNm | 허가번호구분코드명 | |
| hoCnt | 호수(호) | |
| engrGrade | 에너지효율등급 | |
| engrRat | 에너지절감율 | |
| engrEpi | EPI점수 | |
| gnBldGrade | 친환경건축물등급 | |
| gnBldCert | 친환경건축물인증점수 | |
| rnum | 순번 | |
| platPlc | 대지위치 | 예: 서울특별시 강남구 개포동 12번지 |
| sigunguCd | 시군구코드 | 행정표준코드 |
| bjdongCd | 법정동코드 | 행정표준코드 |
| platGbCd | 대지구분코드 | 0:대지, 1:산, 2:블록 |
| bun | 번 | |
| ji | 지 | |
| mgmBldrgstPk | 관리건축물대장PK | 고유 식별자 |
| regstrGbCd | 대장구분코드 | 1:총괄, 2:집합, 3:일반 |
| regstrGbCdNm | 대장구분코드명 | 집합 등 |
| regstrKindCd | 대장종류코드 | |
| regstrKindCdNm | 대장종류코드명 | 총괄표제부 등 |
| newOldRegstrGbCd | 신구대장구분코드 | |
| newOldRegstrGbCdNm | 신구대장구분코드명 | 구대장 등 |
| newPlatPlc | 도로명대지위치 | |
| bldNm | 건물명 | 아파트 단지명 |
| splotNm | 특수지명 | |
| block | 블록 | |
| lot | 로트 | |
| bylotCnt | 외필지수 | |
| naRoadCd | 새주소도로코드 | |
| aptSeq | 단지 고유 식별자 | 국토부 단지코드 |
