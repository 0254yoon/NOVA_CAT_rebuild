
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NOVA-CAT",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}
#MainMenu, footer { visibility: hidden !important; }

:root {
    --bg: #050811;
    --bg2: #09111d;
    --panel: #0d1623;
    --panel2: #111d2b;
    --line: #20354a;
    --cyan: #22d8f4;
    --blue: #2492ff;
    --text: #edf5fb;
    --muted: #8191a6;
    --good: #4fd39a;
    --warn: #ffb44a;
    --danger: #ff6674;
}

html, body, [class*="css"] {
    font-family: "Noto Sans KR", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 78% 0%, rgba(34,216,244,.08), transparent 30rem),
        linear-gradient(180deg, #050811 0%, #07101a 100%);
    color: var(--text);
}
.block-container {
    max-width: 1500px;
    padding-top: .5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #070c14;
    border-right: 1px solid #18283a;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
h1,h2,h3,h4 { color: var(--text) !important; }

.hero {
    background:
        linear-gradient(135deg, rgba(13,42,58,.95), rgba(10,18,31,.98)),
        #0b1420;
    border: 1px solid #1f526b;
    border-radius: 24px;
    padding: 30px 34px;
    margin-bottom: 20px;
    box-shadow: 0 20px 70px rgba(0,0,0,.28);
}
.kicker {
    font-family: "IBM Plex Mono", monospace;
    color: var(--cyan);
    letter-spacing: .16em;
    font-size: .75rem;
    font-weight: 600;
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -.04em;
    margin: 12px 0 8px;
}
.hero-sub {
    color: var(--muted);
    line-height: 1.8;
    max-width: 1050px;
}
.badge {
    display: inline-block;
    padding: 5px 10px;
    margin: 14px 6px 0 0;
    border-radius: 999px;
    border: 1px solid #2a5d76;
    background: #0c2330;
    color: #83ecff;
    font-family: "IBM Plex Mono", monospace;
    font-size: .68rem;
}
.badge.warn {
    border-color: #715020;
    background: #2b1d0a;
    color: #ffc66b;
}

.panel {
    background: linear-gradient(180deg, rgba(17,29,43,.97), rgba(12,20,31,.97));
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    height: 100%;
    box-shadow: 0 14px 42px rgba(0,0,0,.20);
}
.panel.accent {
    border-color: #1f6f89;
}
.label {
    color: var(--cyan);
    font-family: "IBM Plex Mono", monospace;
    font-size: .72rem;
    letter-spacing: .13em;
    font-weight: 600;
}
.value {
    font-family: "IBM Plex Mono", monospace;
    font-size: 1.9rem;
    font-weight: 600;
    margin-top: 8px;
}
.muted { color: var(--muted); }
hr {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #28445c, transparent);
    margin: 18px 0;
}
div[data-testid="stMetric"] {
    background: #0d1623;
    border: 1px solid #20354a;
    border-radius: 16px;
    padding: 15px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid #2388a8;
    background: linear-gradient(135deg, #0b5770, #1687aa);
    color: white;
    font-weight: 700;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #52e6fa;
    color: white;
}
[data-testid="stDataFrame"] {
    border: 1px solid #20354a;
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

DATA = pd.DataFrame([
    {"촉매":"Fe–N–C (Fe–N₄)","FE_CO":98.0,"j_CO":300.0,"TOF":1500.0,"E_ads":-0.65,"d_band":-2.10,"구조":"Fe–N₄"},
    {"촉매":"Pyridinic Ni–N₄–C","FE_CO":99.0,"j_CO":28.3,"TOF":1690.0,"E_ads":1.65,"d_band":-2.50,"구조":"Pyridinic Ni–N₄"},
    {"촉매":"Pyrrolic Ni–N₄–C","FE_CO":94.0,"j_CO":59.6,"TOF":8000.0,"E_ads":1.09,"d_band":-2.00,"구조":"Pyrrolic Ni–N₄"},
    {"촉매":"Ni–N₄–C","FE_CO":97.0,"j_CO":30.6,"TOF":14800.0,"E_ads":1.49,"d_band":-2.30,"구조":"Ni–N₄"},
    {"촉매":"Cu–N–C","FE_CO":96.5,"j_CO":131.3,"TOF":922.0,"E_ads":-2.50,"d_band":-2.48,"구조":"Cu–N–C"},
])

def minmax(s):
    lo, hi = float(s.min()), float(s.max())
    if math.isclose(lo, hi):
        return pd.Series([1.0]*len(s), index=s.index)
    return (s-lo)/(hi-lo)

def gauss(s, target, sigma):
    return np.exp(-((s-target)**2)/(2*max(sigma,1e-9)**2))

def score(df, w, et, es, dt, ds):
    out = df.copy()
    out["S_FE"] = minmax(out["FE_CO"])
    out["S_j"] = minmax(np.log1p(out["j_CO"]))
    out["S_TOF"] = minmax(np.log1p(out["TOF"]))
    out["S_Eads"] = gauss(out["E_ads"], et, es)
    out["S_dband"] = gauss(out["d_band"], dt, ds)
    out["종합점수"] = 100*(
        w["FE"]*out["S_FE"] + w["j"]*out["S_j"] + w["TOF"]*out["S_TOF"] +
        w["Eads"]*out["S_Eads"] + w["dband"]*out["S_dband"]
    )
    return out.sort_values("종합점수", ascending=False).reset_index(drop=True)

def pareto_names(df):
    cols = ["S_FE","S_j","S_TOF","S_Eads","S_dband"]
    arr = df[cols].to_numpy()
    good = []
    for i,row in enumerate(arr):
        dominated = np.any(np.all(arr >= row,axis=1) & np.any(arr > row,axis=1))
        if not dominated:
            good.append(df.iloc[i]["촉매"])
    return good

def sensitivity(df, n, et, es, dt, ds, seed=42):
    rng = np.random.default_rng(seed)
    counts = {c:0 for c in df["촉매"]}
    all_scores = {c:[] for c in df["촉매"]}
    for _ in range(n):
        v = rng.dirichlet(np.ones(5))
        w = {"FE":v[0],"j":v[1],"TOF":v[2],"Eads":v[3],"dband":v[4]}
        s = score(df,w,et,es,dt,ds)
        counts[s.iloc[0]["촉매"]] += 1
        for _,r in s.iterrows():
            all_scores[r["촉매"]].append(r["종합점수"])
    return pd.DataFrame({
        "촉매":counts.keys(),
        "1위 선정 비율(%)":[counts[k]/n*100 for k in counts],
        "평균 점수":[np.mean(all_scores[k]) for k in counts],
        "점수 표준편차":[np.std(all_scores[k]) for k in counts],
    }).sort_values("1위 선정 비율(%)", ascending=False)

st.sidebar.markdown("## NOVA-CAT")
st.sidebar.caption("CO₂ → CO 단원자 촉매 의사결정 플랫폼")

preset = st.sidebar.radio("분석 시나리오",["균형형","선택성 중심","생산성 중심","사용자 지정"])
presets = {
    "균형형":[20,20,20,20,20],
    "선택성 중심":[40,15,10,20,15],
    "생산성 중심":[15,35,30,10,10],
}
defaults = presets.get(preset,[20,20,20,20,20])

st.sidebar.markdown("---")
st.sidebar.markdown("### 평가 가중치")
keys = [("FE","CO 패러데이 효율"),("j","CO 부분 전류밀도"),("TOF","회전 빈도"),("Eads","중간체 흡착에너지"),("dband","d-band center")]
raw={}
for (k,l),d in zip(keys,defaults):
    raw[k] = st.sidebar.slider(l,0,100,int(d),5)
total=sum(raw.values())
w={k:(v/total if total else .2) for k,v in raw.items()}
st.sidebar.caption(f"입력 합계 {total}% · 계산 시 자동 정규화")

st.sidebar.markdown("---")
st.sidebar.markdown("### 적정값 가정")
et = st.sidebar.number_input("흡착에너지 목표값 (eV)",value=-0.65,step=0.05)
es = st.sidebar.number_input("흡착에너지 허용폭 σ",value=1.00,min_value=0.05,step=0.05)
dt = st.sidebar.number_input("d-band center 목표값 (eV)",value=-2.20,step=0.05)
ds = st.sidebar.number_input("d-band 허용폭 σ",value=0.35,min_value=0.05,step=0.05)

st.sidebar.info("흡착에너지와 d-band center는 목표값에 가까울수록 높은 점수를 받습니다.")

st.markdown("""
<div class="hero">
  <div class="kicker">PROJECT NOVA · ENE-02 · CO₂ RR-SAC</div>
  <div class="hero-title">전이금속 단원자 촉매 의사결정 시뮬레이터</div>
  <div class="hero-sub">
    CO 패러데이 효율, 부분 전류밀도, TOF, 중간체 흡착에너지, d-band center를 동시에 반영해
    목적별 최적 촉매를 추천하고, 파레토 분석과 민감도 분석으로 추천 결과의 안정성을 검토합니다.
  </div>
  <span class="badge">MULTI-CRITERIA</span>
  <span class="badge">PARETO FRONT</span>
  <span class="badge">SENSITIVITY TEST</span>
  <span class="badge warn">LITERATURE DATA</span>
</div>
""", unsafe_allow_html=True)

scored = score(DATA,w,et,es,dt,ds)
winner = scored.iloc[0]
pareto = pareto_names(scored)

a,b,c,d = st.columns(4)
a.metric("추천 촉매",winner["촉매"])
b.metric("종합점수",f"{winner['종합점수']:.1f} / 100")
c.metric("파레토 최적 후보",f"{len(pareto)}개")
d.metric("분석 시나리오",preset)

t1,t2,t3,t4 = st.tabs(["01 종합 평가","02 파레토 분석","03 민감도 분석","04 데이터·한계"])

with t1:
    l,r = st.columns([1.08,1])
    with l:
        st.markdown("### 최종 순위")
        show = scored[["촉매","종합점수","FE_CO","j_CO","TOF","E_ads","d_band"]].copy()
        show.columns=["촉매","종합점수","FE_CO (%)","j_CO (mA cm⁻²)","TOF (h⁻¹)","E_ads (eV)","d-band (eV)"]
        st.dataframe(show,use_container_width=True,hide_index=True)
        fig=px.bar(scored.sort_values("종합점수"),x="종합점수",y="촉매",orientation="h",text="종합점수",title="가중치 반영 종합점수")
        fig.update_traces(texttemplate="%{text:.1f}")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#dce8f3",xaxis_gridcolor="#203043",yaxis_title="",height=420)
        st.plotly_chart(fig,use_container_width=True)
    with r:
        strengths={"패러데이 효율":winner["S_FE"],"부분 전류밀도":winner["S_j"],"TOF":winner["S_TOF"],"흡착에너지 적합성":winner["S_Eads"],"d-band 적합성":winner["S_dband"]}
        order=sorted(strengths.items(),key=lambda x:x[1],reverse=True)
        st.markdown(f"""
        <div class="panel accent">
          <div class="label">RECOMMENDED CATALYST</div>
          <div class="value">{winner['촉매']}</div>
          <hr>
          <p><b>주요 강점:</b> {order[0][0]}, {order[1][0]}</p>
          <p><b>상대적 약점:</b> {order[-1][0]}</p>
          <p class="muted">현재 결과는 가중치와 적정값 가정에 따라 달라집니다. 절대적 최고 촉매가 아니라 특정 목적에 가장 적합한 후보로 해석해야 합니다.</p>
        </div>
        """,unsafe_allow_html=True)
        radar = pd.DataFrame({"지표":["FE","j_CO","TOF","E_ads 적합성","d-band 적합성"],"점수":[winner["S_FE"],winner["S_j"],winner["S_TOF"],winner["S_Eads"],winner["S_dband"]]})
        rf=px.line_polar(radar,r="점수",theta="지표",line_close=True,markers=True,range_r=[0,1],title="추천 촉매 성능 프로필")
        rf.update_traces(fill="toself")
        rf.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#dce8f3",polar=dict(bgcolor="rgba(0,0,0,0)"),height=470)
        st.plotly_chart(rf,use_container_width=True)

with t2:
    st.markdown("### 파레토 최적 촉매")
    st.success(" · ".join(pareto))
    fig=px.scatter(scored,x="FE_CO",y="j_CO",size="TOF",color="종합점수",text="촉매",hover_name="촉매",hover_data=["E_ads","d_band"],title="선택성–생산성–TOF 다목적 지도",size_max=42)
    fig.update_traces(textposition="top center")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#dce8f3",xaxis_gridcolor="#203043",yaxis_gridcolor="#203043",height=620)
    st.plotly_chart(fig,use_container_width=True)

with t3:
    st.markdown("### 가중치 민감도 분석")
    n=st.slider("무작위 가중치 조합 수",100,5000,1000,100)
    if st.button("민감도 분석 실행",use_container_width=True):
        with st.spinner("분석 중..."):
            s=sensitivity(DATA,n,et,es,dt,ds)
        st.dataframe(s,use_container_width=True,hide_index=True)
        fig=px.bar(s,x="1위 선정 비율(%)",y="촉매",orientation="h",text="1위 선정 비율(%)",title=f"{n:,}회 무작위 가중치에서의 1위 선정 빈도")
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#dce8f3",xaxis_gridcolor="#203043",height=450)
        st.plotly_chart(fig,use_container_width=True)

with t4:
    st.markdown("### 원자료")
    st.dataframe(DATA,use_container_width=True,hide_index=True)
    st.warning("""
    • 수치는 서로 다른 논문·전위·셀 유형·배위환경에서 얻어졌을 수 있습니다.
    • FE, j_CO, TOF는 클수록 유리하게 처리했습니다.
    • 흡착에너지와 d-band center는 목표값과의 거리를 사용했습니다.
    • Cu–N–C의 FE는 95–98% 범위의 중앙값 96.5%를 사용했습니다.
    • 흡착에너지의 부호와 정의가 문헌마다 다를 수 있으므로 최종 보고서에서 반드시 확인해야 합니다.
    """)
    st.download_button("원자료 CSV 다운로드",DATA.to_csv(index=False).encode("utf-8-sig"),"nova_cat_dataset.csv","text/csv",use_container_width=True)

st.markdown("<div class='muted' style='text-align:center;margin-top:28px;'>NOVA-CAT · Literature-based multi-criteria catalyst decision simulator</div>",unsafe_allow_html=True)
