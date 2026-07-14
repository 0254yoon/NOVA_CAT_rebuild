# NOVA-CAT

CO₂ → CO 전이금속 단원자 촉매 다목적 의사결정 시뮬레이터

## 로컬 실행

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub 업로드

저장소 최상위 폴더에 아래 파일이 바로 보여야 합니다.

- `app.py`
- `requirements.txt`
- `README.md`
- `.streamlit/config.toml`

## Streamlit Community Cloud 배포

1. GitHub 저장소에 파일 업로드
2. Streamlit Community Cloud 접속
3. `New app`
4. 저장소와 branch 선택
5. Main file path에 `app.py` 입력
6. Deploy

## 주요 기능

- 가중치 조절
- 균형형 / 선택성 중심 / 생산성 중심 시나리오
- 종합점수
- 파레토 분석
- 민감도 분석
- 원자료 CSV 다운로드
