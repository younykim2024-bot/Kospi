# KOSPI Wolfe Scanner (모바일용)

이 버전은 **모바일에서 보기 편하게** UI를 바꾼 Streamlit 웹앱입니다.
- 화면 폭이 좁아도 잘 보이도록 `layout="centered"` 적용
- 표/차트는 탭으로 분리
- 차트는 Plotly 캔들(핀치줌/드래그 가능)

## 1) 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2) 모바일에서 쓰는 방법(권장: 무료 배포)
### Streamlit Community Cloud
1) 이 폴더를 GitHub에 올리기
2) Streamlit Cloud에서 repo 연결
3) Deploy 하면 URL이 생깁니다
4) 모바일에서 그 URL로 접속

## 3) 홈화면에 바로가기 추가(안드로이드 크롬)
- 웹앱 열기 → 메뉴(⋮) → '홈 화면에 추가'
