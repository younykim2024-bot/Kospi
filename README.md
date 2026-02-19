# KOSPI Wolfe (Cloud + KOSPI200)

## 핵심
- Streamlit Cloud에서 KRX(data.krx.co.kr) 호출이 Access Denied로 막히는 경우가 있어,
  **KRX를 호출하지 않고** KOSPI200 구성종목을 쓰도록 구성했습니다.
- `kospi200.csv`가 있으면 그것을 기본 universe로 사용합니다.
- `kospi200.csv`가 비어있거나 없으면, 네이버 금융 `entryJongmok` 페이지에서 구성종목을 크롤링해 생성합니다.

## 참고(근거)
네이버 금융에서 KOSPI200 편입종목 테이블을 페이지별로 제공하며,
`entryJongmok.nhn?page=...` 형태로 접근해 종목명/코드를 파싱하는 방식이 널리 사용됩니다.
