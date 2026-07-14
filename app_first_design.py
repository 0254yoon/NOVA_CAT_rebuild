
import math
import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NOVA-CAT | CO₂RR SAC Decision Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme / visual design
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');


[data-testid="stHeader"] {display:none;}
[data-testid="stToolbar"] {display:none;}
[data-testid="stDecoration"] {display:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

:root {
    --bg: #050812;
    --panel: #0e1420;
    --panel2: #121b29;
    --line: #23354a;
    --cyan: #20d6f2;
    --blue: #238bff;
    --text: #eef5fb;
    --muted: #8392a8;
    --good: #54d39b;
    --warn: #ffb347;
    --bad: #ff6672;
}

html, body, [class*="css"] {
    font-family: "Noto Sans KR", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 75% 0%, rgba(32, 214, 242, 0.07), transparent 24rem),
        linear-gradient(180deg, #050812 0%, #070b14 100%);
    color: var(--text);
}
.block-container {
    max-width: 1500px;
    padding-top: 0.4rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #070b13;
    border-right: 1px solid #162235;
}
[data-testid="stSidebar"] * {
    color: #dfe9f5;
}
h1, h2, h3 {
    color: var(--text) !important;
}
.mono {
    font-family: "IBM Plex Mono", monospace;
}
.hero {
    border: 1px solid #1c3850;
    background:
        linear-gradient(135deg, rgba(13, 38, 54, 0.92), rgba(13, 18, 31, 0.97)),
        #0d1420;
    border-radius: 24px;
    padding: 30px 32px;
    margin-bottom: 20px;
    box-shadow: 0 18px 60px rgba(0, 0, 0, .32);
}
.kicker {
    color: var(--cyan);
    font-family: "IBM Plex Mono", monospace;
    font-size: .78rem;
    letter-spacing: .15em;
    font-weight: 600;
}
.hero-title {
    margin: 12px 0 8px;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -.03em;
}
.hero-sub {
    color: var(--muted);
    line-height: 1.7;
}
.card {
    background: linear-gradient(180deg, rgba(18, 27, 41, 0.97), rgba(13, 20, 32, 0.97));
    border: 1px solid var(--line);
    border-radius: 19px;
    padding: 20px;
    height: 100%;
    box-shadow: 0 12px 34px rgba(0,0,0,.20);
}
.card-accent {
    border-color: #1c6a82;
    box-shadow: inset 0 0 0 1px rgba(32,214,242,.05), 0 12px 40px rgba(0,0,0,.25);
}
.small-label {
    color: var(--cyan);
    font-family: "IBM Plex Mono", monospace;
    font-size: .72rem;
    letter-spacing: .12em;
    font-weight: 600;
}
.big-number {
    font-family: "IBM Plex Mono", monospace;
    font-size: 2rem;
    font-weight: 600;
    margin-top: 8px;
}
.muted {
    color: var(--muted);
}
.badge {
    display:inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    border: 1px solid #28516a;
    background: #0d2230;
    color: #7feaff;
    font-family: "IBM Plex Mono", monospace;
    font-size: .70rem;
    margin-right: 6px;
}
.badge-warn {
    border-color: #664a23;
    background: #2a1d0d;
    color: #ffc777;
}
.rule {
    height: 1px;
    background: linear-gradient(90deg, transparent, #27425a, transparent);
    margin: 18px 0;
}
div[data-testid="stMetric"] {
    background: #0e1622;
    border: 1px solid #203246;
    padding: 16px;
    border-radius: 16px;
}
.stButton > button {
    border-radius: 12px;
    border: 1px solid #1f85a3;
    background: linear-gradient(135deg, #0b4f67, #117ca0);
    color: white;
    font-weight: 700;
}
.stButton > button:hover {
    border-color: #43dff6;
    color: white;
}
[data-baseweb="slider"] {
    padding-top: .25rem;
}
[data-testid="stDataFrame"] {
    border: 1px solid #213248;
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Dataset
# -----------------------------
data = pd.DataFrame([
    {
        "촉매": "Fe–N–C (Fe–N₄)",
        "FE_CO": 98.0,
        "j_CO": 300.0,
        "TOF": 1500.0,
        "E_ads": -0.65,
        "d_band": -2.10,
        "구조": "Fe–N₄",
        "메모": "비교용 대표값"
    },
    {
        "촉매": "Pyridinic Ni–N₄–C",
        "FE_CO": 99.0,
        "j_CO": 28.3,
        "TOF": 1690.0,
        "E_ads": 1.65,
        "d_band": -2.50,
        "구조": "Pyridinic Ni–N₄",
        "메모": "Ni 배위구조 비교"
    },
    {
        "촉매": "Pyrrolic Ni–N₄–C",
        "FE_CO": 94.0,
        "j_CO": 59.6,
        "TOF": 8000.0,
        "E_ads": 1.09,
        "d_band": -2.00,
        "구조": "Pyrrolic Ni–N₄",
        "메모": "Ni 배위구조 비교"
    },
    {
        "촉매": "Ni–N₄–C",
        "FE_CO": 97.0,
        "j_CO": 30.6,
        "TOF": 14800.0,
        "E_ads": 1.49,
        "d_band": -2.30,
        "구조": "Ni–N₄",
        "메모": "대표 Ni–N₄"
    },
    {
        "촉매": "Cu–N–C",
        "FE_CO": 96.5,
        "j_CO": 131.3,
        "TOF": 922.0,
        "E_ads": -2.50,
        "d_band": -2.48,
        "구조": "Cu–N–C",
        "메모": "FE 95–98%의 중앙값 사용"
    },
])

# -----------------------------
# Utility functions
# -----------------------------
def minmax(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if math.isclose(lo, hi):
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)

def gaussian_preference(series: pd.Series, target: float, sigma: float) -> pd.Series:
    sigma = max(float(sigma), 1e-9)
    return np.exp(-((series - target) ** 2) / (2 * sigma ** 2))

def calculate_scores(df, weights, e_target, e_sigma, d_target, d_sigma):
    scored = df.copy()
    scored["S_FE"] = minmax(scored["FE_CO"])
    scored["S_j"] = minmax(np.log1p(scored["j_CO"]))
    scored["S_TOF"] = minmax(np.log1p(scored["TOF"]))
    scored["S_Eads"] = gaussian_preference(scored["E_ads"], e_target, e_sigma)
    scored["S_dband"] = gaussian_preference(scored["d_band"], d_target, d_sigma)

    scored["종합점수"] = 100 * (
        weights["FE"] * scored["S_FE"] +
        weights["j"] * scored["S_j"] +
        weights["TOF"] * scored["S_TOF"] +
        weights["Eads"] * scored["S_Eads"] +
        weights["dband"] * scored["S_dband"]
    )
    return scored.sort_values("종합점수", ascending=False).reset_index(drop=True)

def pareto_front(df):
    # All score columns are already "higher is better"
    cols = ["S_FE", "S_j", "S_TOF", "S_Eads", "S_dband"]
    values = df[cols].to_numpy()
    efficient = np.ones(len(df), dtype=bool)
    for i, row in enumerate(values):
        if not efficient[i]:
            continue
        dominated_by_any = np.any(
            np.all(values >= row, axis=1) & np.any(values > row, axis=1)
        )
        efficient[i] = not dominated_by_any
    return df.loc[efficient, "촉매"].tolist()

def run_sensitivity(df, n, e_target, e_sigma, d_target, d_sigma, seed=42):
    rng = np.random.default_rng(seed)
    counts = {name: 0 for name in df["촉매"]}
    avg_scores = {name: [] for name in df["촉매"]}
    for _ in range(n):
        w = rng.dirichlet(np.ones(5))
        weights = dict(FE=w[0], j=w[1], TOF=w[2], Eads=w[3], dband=w[4])
        scored = calculate_scores(df, weights, e_target, e_sigma, d_target, d_sigma)
        counts[scored.iloc[0]["촉매"]] += 1
        for _, row in scored.iterrows():
            avg_scores[row["촉매"]].append(row["종합점수"])
    out = pd.DataFrame({
        "촉매": list(counts.keys()),
        "1위 선정 비율(%)": [counts[k] / n * 100 for k in counts],
        "평균 점수": [np.mean(avg_scores[k]) for k in counts],
        "점수 표준편차": [np.std(avg_scores[k]) for k in counts],
    })
    return out.sort_values("1위 선정 비율(%)", ascending=False)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## NOVA-CAT")
st.sidebar.caption("CO₂ → CO 단원자 촉매 다목적 의사결정 플랫폼")

preset = st.sidebar.radio(
    "분석 시나리오",
    ["균형형", "선택성 중심", "생산성 중심", "사용자 지정"],
    index=0,
)

preset_values = {
    "균형형": [20, 20, 20, 20, 20],
    "선택성 중심": [40, 15, 10, 20, 15],
    "생산성 중심": [15, 35, 30, 10, 10],
}
default_weights = preset_values.get(preset, [20, 20, 20, 20, 20])

st.sidebar.markdown("---")
st.sidebar.markdown("### 평가 가중치")

labels = [
    ("FE", "CO 패러데이 효율"),
    ("j", "CO 부분 전류밀도"),
    ("TOF", "회전 빈도"),
    ("Eads", "중간체 흡착 에너지"),
    ("dband", "d-band center"),
]

raw = {}
for (key, label), default in zip(labels, default_weights):
    raw[key] = st.sidebar.slider(label, 0, 100, int(default), 5)

total = sum(raw.values())
if total == 0:
    st.sidebar.error("가중치 합이 0입니다.")
    normalized = {k: 0.2 for k in raw}
else:
    normalized = {k: v / total for k, v in raw.items()}
st.sidebar.caption(f"입력 가중치 합: {total}% · 계산 시 자동 정규화")

st.sidebar.markdown("---")
st.sidebar.markdown("### 적정값 가정")
e_target = st.sidebar.number_input("흡착에너지 목표값 (eV)", value=-0.65, step=0.05)
e_sigma = st.sidebar.number_input("흡착에너지 허용폭 σ", value=1.00, min_value=0.05, step=0.05)
d_target = st.sidebar.number_input("d-band center 목표값 (eV)", value=-2.20, step=0.05)
d_sigma = st.sidebar.number_input("d-band 허용폭 σ", value=0.35, min_value=0.05, step=0.05)

st.sidebar.info(
    "흡착에너지와 d-band center는 무조건 크거나 작을수록 좋은 지표가 아니므로 "
    "목표값과의 거리를 기준으로 점수화합니다."
)

# -----------------------------
# Main UI
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="kicker">PROJECT NOVA · ENE-02 · CO₂RR-SAC</div>
    <div class="hero-title">전이금속 단원자 촉매 의사결정 시뮬레이터</div>
    <div class="hero-sub">
        패러데이 효율, CO 부분 전류밀도, TOF, 중간체 흡착에너지, d-band center를
        동시에 반영하여 목적별 최적 촉매를 추천하고, 파레토 분석과 민감도 분석으로
        추천 결과의 안정성을 검토합니다.
    </div>
    <div style="margin-top:15px">
        <span class="badge">MULTI-CRITERIA</span>
        <span class="badge">PARETO FRONT</span>
        <span class="badge">SENSITIVITY TEST</span>
        <span class="badge badge-warn">LITERATURE DATA</span>
    </div>
</div>
""", unsafe_allow_html=True)

scored = calculate_scores(data, normalized, e_target, e_sigma, d_target, d_sigma)
winner = scored.iloc[0]
pareto = pareto_front(scored)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("추천 촉매", winner["촉매"])
with c2:
    st.metric("종합점수", f"{winner['종합점수']:.1f} / 100")
with c3:
    st.metric("파레토 최적 후보", f"{len(pareto)}개")
with c4:
    st.metric("분석 시나리오", preset)

tab1, tab2, tab3, tab4 = st.tabs(
    ["01 종합 평가", "02 파레토 분석", "03 민감도 분석", "04 데이터·한계"]
)

with tab1:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### 최종 순위")
        display = scored[["촉매", "종합점수", "FE_CO", "j_CO", "TOF", "E_ads", "d_band"]].copy()
        display.columns = ["촉매", "종합점수", "FE_CO (%)", "j_CO (mA cm⁻²)", "TOF (h⁻¹)", "E_ads (eV)", "d-band (eV)"]
        st.dataframe(
            display.style.format({
                "종합점수": "{:.1f}",
                "FE_CO (%)": "{:.1f}",
                "j_CO (mA cm⁻²)": "{:.1f}",
                "TOF (h⁻¹)": "{:.0f}",
                "E_ads (eV)": "{:.2f}",
                "d-band (eV)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        bar = px.bar(
            scored.sort_values("종합점수"),
            x="종합점수",
            y="촉매",
            orientation="h",
            text=scored.sort_values("종합점수")["종합점수"].round(1),
            title="가중치 반영 종합점수",
        )
        bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dbe7f3",
            xaxis_gridcolor="#203043",
            yaxis_title="",
            xaxis_title="Score",
            height=420,
        )
        st.plotly_chart(bar, use_container_width=True)

    with right:
        st.markdown("### 추천 근거")
        best_factors = {
            "패러데이 효율": winner["S_FE"],
            "부분 전류밀도": winner["S_j"],
            "TOF": winner["S_TOF"],
            "흡착에너지 적합성": winner["S_Eads"],
            "d-band 적합성": winner["S_dband"],
        }
        best_sorted = sorted(best_factors.items(), key=lambda x: x[1], reverse=True)
        strongest = ", ".join([x[0] for x in best_sorted[:2]])
        weakest = ", ".join([x[0] for x in best_sorted[-2:]])
        st.markdown(
            f"""
            <div class="card card-accent">
                <div class="small-label">RECOMMENDED CATALYST</div>
                <div class="big-number">{winner['촉매']}</div>
                <div class="rule"></div>
                <p><b>강점:</b> {strongest}</p>
                <p><b>상대적 약점:</b> {weakest}</p>
                <p class="muted">
                    현재 결과는 사용자가 설정한 가중치와 적정값 가정에 따라 달라집니다.
                    따라서 절대적인 '최고 촉매'라기보다 특정 목적에 가장 적합한 후보로 해석해야 합니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_cols = ["S_FE", "S_j", "S_TOF", "S_Eads", "S_dband"]
        radar_df = winner[score_cols].to_frame().reset_index()
        radar_df.columns = ["지표", "점수"]
        radar_df["지표"] = ["FE", "j_CO", "TOF", "E_ads 적합성", "d-band 적합성"]
        radar = px.line_polar(
            radar_df,
            r="점수",
            theta="지표",
            line_close=True,
            markers=True,
            range_r=[0, 1],
            title="추천 촉매 성능 프로필",
        )
        radar.update_traces(fill="toself")
        radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(gridcolor="#274057", tickfont_color="#8da0b5"),
                angularaxis=dict(gridcolor="#274057", tickfont_color="#dce8f3"),
            ),
            font_color="#dce8f3",
            height=470,
        )
        st.plotly_chart(radar, use_container_width=True)

with tab2:
    st.markdown("### 파레토 최적 촉매")
    st.caption("다른 지표를 악화시키지 않고서는 어느 한 지표도 더 개선할 수 없는 후보입니다.")
    st.success(" · ".join(pareto))

    scatter = px.scatter(
        scored,
        x="FE_CO",
        y="j_CO",
        size="TOF",
        color="종합점수",
        hover_name="촉매",
        hover_data={"E_ads": True, "d_band": True, "종합점수": ":.1f"},
        text="촉매",
        title="선택성–생산성–TOF 다목적 지도",
        size_max=42,
    )
    scatter.update_traces(textposition="top center")
    scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce8f3",
        xaxis_gridcolor="#203043",
        yaxis_gridcolor="#203043",
        xaxis_title="CO Faradaic efficiency (%)",
        yaxis_title="CO partial current density (mA cm⁻²)",
        height=620,
    )
    st.plotly_chart(scatter, use_container_width=True)

with tab3:
    st.markdown("### 가중치 민감도 분석")
    n_runs = st.slider("무작위 가중치 조합 수", 100, 5000, 1000, 100)
    if st.button("민감도 분석 실행", use_container_width=True):
        with st.spinner("다양한 가중치 조합을 평가하는 중입니다..."):
            sensitivity = run_sensitivity(
                data, n_runs, e_target, e_sigma, d_target, d_sigma
            )
        st.dataframe(
            sensitivity.style.format({
                "1위 선정 비율(%)": "{:.1f}",
                "평균 점수": "{:.1f}",
                "점수 표준편차": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        fig = px.bar(
            sensitivity,
            x="1위 선정 비율(%)",
            y="촉매",
            orientation="h",
            text=sensitivity["1위 선정 비율(%)"].round(1),
            title=f"{n_runs:,}회 무작위 가중치에서의 1위 선정 빈도",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce8f3",
            xaxis_gridcolor="#203043",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 원자료")
    st.dataframe(data, use_container_width=True, hide_index=True)

    st.markdown("### 해석상 주의")
    st.warning(
        """
        1. 각 수치는 서로 다른 논문·전위·셀 유형·배위환경에서 얻어졌을 수 있으므로 절대적 직접 비교에는 제한이 있습니다.
        2. FE, j_CO, TOF는 클수록 유리하게 처리했지만, 흡착에너지와 d-band center는 사바티에 원리에 따라 목표값과의 거리를 사용했습니다.
        3. Cu–N–C의 FE는 95–98% 범위의 중앙값 96.5%를 사용했습니다.
        4. 입력된 흡착에너지 값은 부호와 정의가 문헌마다 다를 수 있으므로, 최종 보고서에서는 반드시 동일한 정의인지 재확인해야 합니다.
        """
    )

    st.download_button(
        "현재 원자료 CSV 다운로드",
        data.to_csv(index=False).encode("utf-8-sig"),
        file_name="nova_cat_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    "<div class='muted' style='text-align:center; margin-top:28px;'>"
    "NOVA-CAT · Literature-based multi-criteria catalyst decision simulator"
    "</div>",
    unsafe_allow_html=True,
)
