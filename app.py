import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import google.generativeai as genai

# --- (0) 페이지 설정 ---
st.set_page_config(
    page_title="결혼이주여성 정신건강 대시보드",
    page_icon="🏥",
    layout="wide"
)

# --- (1) 데이터베이스 및 분석 결과 상수 ---
# app.py와 같은 폴더의 multicultural.db를 읽는다(로컬·Streamlit Cloud 공통).
# 이 앱은 원본 CSV를 직접 읽지 않으므로, 레포에 multicultural.db만 있으면 동작한다.
_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "multicultural.db")
if not os.path.exists(DB_PATH):  # 혹시 작업 폴더 기준으로 실행될 때의 대비
    DB_PATH = os.path.join(os.getcwd(), "multicultural.db")

# [초기 가설] 지역 밀집도 모형 — 설명력이 약해 심화분석의 출발점이 됨
ANALYSIS_RESULTS = {
    "H1": {"coef_region": 0.0349, "p_val": 0.001, "r2": 0.0277,
           "decision": "채택 (통계적 유의하나 효과 미약, R²=2.8%)"},
    "H2": {"high_avg": 3.204, "low_avg": 3.210, "t_stat": -0.343, "p_val": 0.732,
           "decision": "기각"},
}

# [심화분석] 다문화수용성조사 2021 (n=5,000) — '시선'을 만드는 진짜 요인
#  · 종속변수: 배타적 태도 지수(15문항, 1~6점, Cronbach's α=0.871)
DEEP = {
    # 배타적 태도와의 상관계수 (Pearson r) — 큰 값일수록 배타성과 강하게 연결
    "corr": [
        ("위협 인식(일자리·범죄·재정)", 0.367, True),
        ("단일민족주의", 0.253, True),
        ("다문화지향성", -0.186, True),
        ("연령(고령)", 0.143, True),
        ("학력(고학력)", -0.115, True),
        ("부정 미디어 노출", 0.092, True),
        ("이주민 친구 수(접촉)", -0.048, False),
        ("가구소득", -0.028, False),
    ],
    "reg": {"r2": 0.180, "n": 4885,
            "beta": [("위협 인식", 0.262, True), ("단일민족주의", 0.155, True),
                     ("다문화지향성", -0.119, True), ("연령", 0.004, True)]},
    "generation": {"old_avg": 3.306, "old_n": 1578, "young_avg": 3.018,
                   "young_n": 431, "t": 7.78, "p": "<0.001"},
    "abstract_gap": {"abstract": 38.2, "concrete": 39.3},  # %, 동의(4점 이상)
}


# --- (2) 데이터 로드 함수 ---
@st.cache_data
def run_query(query):
    if not os.path.exists(DB_PATH):
        st.error(f"multicultural.db를 찾을 수 없습니다. app.py와 같은 폴더(레포 루트)에 DB 파일을 올렸는지 확인하세요. (경로: {DB_PATH})")
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")
        return pd.DataFrame()


# --- (3) 사이드바 네비게이션 ---
st.sidebar.title("📌 바로가기")
sections = ["데이터 개요", "인구통계 현황", "가설 검증 분석",
            "SQL 쿼리 설명", "인사이트 & AI 브리핑"]
selection = st.sidebar.radio("섹션을 선택하세요", sections)

# --- (A) 헤더 + 핵심 지표 카드 (모든 섹션 공통) ---
st.title("결혼이주여성 14만 시대, 왜 10명 중 1명은 죽음을 생각하는가?")
st.markdown("#### 한국 사회의 '차가운 시선'과 이주여성의 '심리적 고립' 사이의 구조적 미스매치")

col1, col2, col3, col4 = st.columns(4)
col1.metric("결혼이민자 총수(2024)", "181,436명")
col2.metric("여성 비율", "80.3%", "145,731명")
col3.metric("우울장애 유병률", "8.3%")
col4.metric("자살생각 경험률", "12.9%", "10명 중 1명", delta_color="inverse")
st.divider()

# ============================================================
# (B) 데이터 개요
# ============================================================
if selection == "데이터 개요":
    st.subheader("📊 데이터 구성 및 출처 (11개 테이블)")
    data_desc = [
        {"분류": "당사자 실태", "Table": "mental_health_overall", "내용": "우울·불안·자살생각 유병률", "출처": "보사연·질병청(2024, n=519)"},
        {"분류": "당사자 실태", "Table": "mental_health_by_characteristic", "내용": "특성별(★차별경험) 정신건강", "출처": "보사연·질병청(2024)"},
        {"분류": "당사자 실태", "Table": "suicide_reason", "내용": "자살생각 원인(복수응답)", "출처": "보사연·질병청(2024)"},
        {"분류": "당사자 실태", "Table": "mental_health_care", "내용": "상담 인지 및 이용 경험", "출처": "보사연·질병청(2024)"},
        {"분류": "사회적 인식", "Table": "perception_immigrant", "내용": "한국인의 이주민 인식·우려", "출처": "KOSSDA(2024, n=1,000)"},
        {"분류": "거시 지표", "Table": "immigrant_trend_total", "내용": "연도별 결혼이민자 총수", "출처": "법무부 통계연보"},
        {"분류": "거시 지표", "Table": "immigrant_by_gender", "내용": "연도별 성별 분포", "출처": "법무부 통계연보"},
        {"분류": "거시 지표", "Table": "immigrant_by_nationality", "내용": "연도별 국적별 분포", "출처": "법무부 통계연보"},
        {"분류": "지역 인프라(JOIN)", "Table": "pop_female_sido", "내용": "시도별 여성 결혼이민자 수", "출처": "행정안전부(2024)"},
        {"분류": "지역 인프라(JOIN)", "Table": "infra_center_sido", "내용": "시도별 다문화가족지원센터", "출처": "한국건강가정진흥원(2025)"},
        {"분류": "지역 인프라(JOIN)", "Table": "infra_korean_edu_sido", "내용": "시도별 한국어교육기관", "출처": "여성가족부(2025)"},
    ]
    st.dataframe(pd.DataFrame(data_desc), use_container_width=True, hide_index=True)
    st.caption("※ 가설 검증용 개인 원자료: 2021 국민 다문화수용성조사(n=5,000)를 별도 활용")

# ============================================================
# (C) 인구통계 현황
# ============================================================
elif selection == "인구통계 현황":
    st.subheader("📈 결혼이주여성 인구통계 현황")
    c_pop, c_trend = st.columns(2)

    with c_pop:
        st.write("##### 지역(시도)별 여성 결혼이민자 수")
        df_pop = run_query("SELECT * FROM pop_female_sido ORDER BY female_marriage_immigrants DESC")
        if not df_pop.empty:
            x = 'sido' if 'sido' in df_pop.columns else df_pop.columns[0]
            y = 'female_marriage_immigrants' if 'female_marriage_immigrants' in df_pop.columns else df_pop.columns[1]
            fig = px.bar(df_pop, x=x, y=y, text_auto='.2s', color=y, color_continuous_scale='Blues')
            fig.update_layout(xaxis_title="시도", yaxis_title="여성 결혼이민자 수", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with c_trend:
        st.write("##### 연도별 결혼이민자 증가 추이")
        df_trend = run_query("SELECT * FROM immigrant_trend_total ORDER BY year")
        if not df_trend.empty:
            x = 'year' if 'year' in df_trend.columns else df_trend.columns[0]
            y = 'total_count' if 'total_count' in df_trend.columns else df_trend.columns[1]
            fig = px.area(df_trend, x=x, y=y, markers=True)
            fig.update_layout(xaxis_title="연도", yaxis_title="총 결혼이민자 수")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c_gen, c_nat = st.columns(2)
    with c_gen:
        st.write("##### 연도별 성별 분포")
        df_g = run_query("SELECT * FROM immigrant_by_gender ORDER BY year")
        if not df_g.empty:
            fig = px.bar(df_g, x='year', y='n', color='gender', barmode='stack')
            fig.update_layout(xaxis_title="연도", yaxis_title="명")
            st.plotly_chart(fig, use_container_width=True)
    with c_nat:
        st.write("##### 국적별 분포 (2024)")
        df_n = run_query("SELECT * FROM immigrant_by_nationality WHERE year=2024 ORDER BY n DESC")
        if not df_n.empty:
            fig = px.bar(df_n, x='nationality', y='n', color='n', color_continuous_scale='Tealgrn')
            fig.update_layout(xaxis_title="국적", yaxis_title="명", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# (D) 가설 검증 분석 — 핵심
# ============================================================
elif selection == "가설 검증 분석":
    st.subheader("🧪 분석의 전환: '지역' 가설에서 '사회 전체' 발견으로")

    # --- 초기 가설(지역 밀집도) → 약함 ---
    st.markdown("##### ① 초기 가설: 이주여성이 많은 지역일수록 시선이 차가운가?")
    c1, c2 = st.columns(2)
    c1.info(f"**H1 (집단위협)** 지역 내 여성 이민자 수↑ → 배타성↑?\n\n"
            f"**결과: {ANALYSIS_RESULTS['H1']['decision']}**")
    c2.info(f"**H2** 밀집 상위 vs 하위 지역 배타성 차이?\n\n"
            f"**결과: 기각** (t={ANALYSIS_RESULTS['H2']['t_stat']}, p={ANALYSIS_RESULTS['H2']['p_val']})")
    st.warning("➡️ 지역 밀집도의 설명력은 **2.8%**에 불과. "
               "차가운 시선은 '특정 지역'의 문제가 아니라는 뜻 → 무엇이 진짜 요인인지 심화 분석.")

    st.divider()
    st.markdown("##### ② 심화분석: '차가운 시선'을 만드는 진짜 요인 (n=5,000)")

    cc1, cc2 = st.columns([1.2, 1])
    with cc1:
        # 상관계수 막대 (배타성과의 상관)
        corr_df = pd.DataFrame(DEEP["corr"], columns=["요인", "상관계수", "유의"])
        corr_df["방향"] = corr_df["상관계수"].apply(lambda r: "배타성↑" if r > 0 else "수용성↑")
        fig = px.bar(corr_df.sort_values("상관계수"), x="상관계수", y="요인",
                     orientation="h", color="방향",
                     color_discrete_map={"배타성↑": "#d62728", "수용성↑": "#1f77b4"},
                     title="배타적 태도와의 상관계수 (Pearson r)")
        fig.add_vline(x=0, line_width=1, line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("회귀 모형 설명력 R² = 0.180 (지역 모형 0.028의 6배 이상), n=4,885")
    with cc2:
        st.markdown(
            "**핵심 동력 = 위협 인식**\n\n"
            "“일자리를 뺏고·범죄율을 높이고·재정 부담을 키운다”는 위협 프레임이 "
            "배타성과 가장 강하게 결합(r=0.37).\n\n"
            "**단일민족주의**(혈통주의)도 강한 정적 요인, **다문화지향성**은 보호 요인.\n\n"
            "⚠️ **이주민 친구 수(접촉)는 무관(비유의)** — 단순히 곁에 두는 것만으로는 "
            "시선이 바뀌지 않음(단순접촉가설 미지지).")

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.write("##### ③ 세대 격차: 고령일수록 배타적")
        gen = DEEP["generation"]
        fig = go.Figure(go.Bar(
            x=["20대 이하", "60대 이상"], y=[gen["young_avg"], gen["old_avg"]],
            marker_color=["#1f77b4", "#d62728"],
            text=[f'{gen["young_avg"]:.2f}', f'{gen["old_avg"]:.2f}'], textposition="outside"))
        fig.update_layout(yaxis_title="배타적 태도(1~6점)", yaxis_range=[2.5, 3.6])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"t={gen['t']}, p{gen['p']} (통계적으로 유의한 세대 차이)")
    with g2:
        st.write("##### ④ 관용의 추상성: 말과 현실의 격차")
        gap = DEEP["abstract_gap"]
        fig = go.Figure(go.Bar(
            x=["“다양성 늘수록 좋다”<br>(원칙적 찬성)", "“내 이웃은 싫다”<br>(구체적 거부)"],
            y=[gap["abstract"], gap["concrete"]],
            marker_color=["#2ca02c", "#d62728"],
            text=[f'{gap["abstract"]}%', f'{gap["concrete"]}%'], textposition="outside"))
        fig.update_layout(yaxis_title="동의율(%)", yaxis_range=[0, 50])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("원칙적 찬성과 내 이웃 거부가 거의 동률 → 관용이 추상적 수준에 머묾")

    st.divider()
    # --- 결과(고통) 측: 차별 → 정신건강 ---
    st.subheader("🚨 그 시선의 결과: 차별이 정신건강을 무너뜨린다")
    df_all = run_query("SELECT * FROM mental_health_by_characteristic")
    if not df_all.empty:
        cat = 'char_category' if 'char_category' in df_all.columns else df_all.columns[1]
        grp = 'char_group' if 'char_group' in df_all.columns else df_all.columns[0]
        dep = 'depression_pct' if 'depression_pct' in df_all.columns else df_all.columns[2]
        sui = 'suicide_pct' if 'suicide_pct' in df_all.columns else df_all.columns[4]
        df_d = df_all[df_all[grp].astype(str).str.contains('차별')]
        if not df_d.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_d[cat], y=df_d[sui], name='자살생각(%)', marker_color="#d62728"))
            fig.add_trace(go.Bar(x=df_d[cat], y=df_d[dep], name='우울장애(%)', marker_color="#ff9896"))
            fig.update_layout(title="차별 경험 유무에 따른 정신건강 지표 (자살생각 30.6% vs 6.1%, 약 5배)",
                              barmode='group', xaxis_title="차별 경험", yaxis_title="비율(%)")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 그러나 돌봄은 닿지 않는다 (인지→상담 누수)")
    funnel = dict(number=[100, 88.3, 9.8, 3.2],
                  stage=["전체 응답자", "정신건강 중요성 인지", "상담 필요성 인지", "실제 상담 이용"])
    fig = px.funnel(funnel, x='number', y='stage', title="도움이 필요해도 닿지 못하는 구조")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# (E) SQL 쿼리 설명
# ============================================================
elif selection == "SQL 쿼리 설명":
    st.subheader("🔍 주요 SQL 분석 로직")
    with st.expander("① 시도(sido) 키로 인구·인프라 결합 — 지역 미스매치", expanded=True):
        st.code("""SELECT p.sido, p.female_marriage_immigrants AS 여성인구,
       c.center_count AS 지원센터, e.org_count AS 한국어교육기관
FROM pop_female_sido p
LEFT JOIN infra_center_sido     c ON p.sido = c.sido
LEFT JOIN infra_korean_edu_sido e ON p.sido = e.sido
ORDER BY 여성인구 DESC;""", language='sql')
        st.write("지역별 이주여성 인구 대비 지원 인프라의 불균형을 보는 JOIN 쿼리입니다.")
    with st.expander("② 분석용 결합 — 개인(수용성조사) ⋈ 지역 인구"):
        st.code("""SELECT a.*, p.female_marriage_immigrants AS region_women
FROM acceptance a
JOIN pop_female_sido p ON a.sido = p.sido;""", language='sql')
        st.write("개인 응답에 거주 지역 변수를 붙여 초기 가설(H1)을 검증했습니다.")

# ============================================================
# (F) 인사이트 & AI 브리핑
# ============================================================
elif selection == "인사이트 & AI 브리핑":
    st.subheader("💡 핵심 인사이트")
    i1, i2, i3 = st.columns(3)
    i1.error("**1. 차별은 치명적 균열**\n\n차별 경험 시 자살생각이 **5배**(30.6% vs 6.1%). 정신건강을 가르는 단일 최대 요인.")
    i2.warning("**2. 시선은 '위협 인식'에서 온다**\n\n지역 밀집도가 아니라 위협 인식·혈통주의·세대가 배타성을 만든다(R² 2.8%→18%). 단순 접촉으론 안 바뀐다.")
    i3.info("**3. 관용의 추상성 + 돌봄 공백**\n\n원칙엔 찬성(38%)하나 내 이웃은 거부(39%). 상담 이용은 3.2%까지 누수.")

    st.divider()
    st.subheader("🤖 Gemini AI 정책 브리핑 생성")
    st.caption("※ API 키는 코드에 넣지 말고 secrets로만 관리 — 로컬: .streamlit/secrets.toml / 배포: Streamlit Cloud ▸ Settings ▸ Secrets")

    # 시크릿 안전 조회 (secrets 파일이 아예 없어도 앱이 죽지 않도록)
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None

    if st.button("AI 브리핑 생성하기", type="primary"):
        if not api_key:
            st.warning("Gemini API 키가 설정되지 않았습니다. `.streamlit/secrets.toml`에 GEMINI_API_KEY를 추가하세요.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')  # 'gemini-pro'는 단종 → 최신 모델 사용
                prompt = """당신은 보건복지부 정책 보좌관입니다. 아래 데이터 근거로 결혼이주여성 정신건강 정책 제언을 3~5문장으로 작성하세요.
- 여성 결혼이민자 14.6만명(전체의 80.3%), 자살생각 경험 12.9%
- 차별 경험 시 자살생각 5배 증가(30.6% vs 6.1%)
- 한국인의 배타적 태도는 '위협 인식·혈통주의·고령'이 핵심 동력(회귀 R²=0.18), 단순 접촉으론 완화되지 않음
- 원칙적 다양성 찬성(38%)과 '내 이웃 거부'(39%)가 거의 동률 — 관용이 추상적 수준에 머묾
- 인프라는 한국어 교육에 편중, 실제 상담 이용은 3% 수준으로 매우 낮음"""
                response = model.generate_content(prompt)
                st.success("AI 정책 브리핑")
                st.write(response.text)
            except Exception as e:
                st.error(f"AI 호출 오류: {e}")

    st.caption("**한계:** 시도 코드 매핑 가정, 상관≠인과, 정신건강 데이터는 집계표 기반.")
