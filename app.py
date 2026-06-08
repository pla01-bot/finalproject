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

# --- (1) 데이터베이스 및 상수 설정 ---
DB_PATH = os.path.join(os.path.dirname(__file__), "multicultural.db")

# 외부 분석 결과 (H1, H2 상수)
ANALYSIS_RESULTS = {
    "H1": {
        "formula": "exclusion_score ~ region_women_10k + age + female + media_contact + C(region_size)",
        "coef_region": 0.0349,
        "p_val": 0.001,
        "age_coef": 0.0069,
        "r2": 0.0277,
        "decision": "채택 (통계적 유의하나 효과 미약)"
    },
    "H2": {
        "high_avg": 3.204,
        "low_avg": 3.210,
        "t_stat": -0.343,
        "p_val": 0.732,
        "decision": "기각"
    }
}

# --- (2) 데이터 로드 함수 ---
@st.cache_data
def run_query(query):
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
sections = {
    "메인": "header",
    "데이터 개요": "data_info",
    "인구통계 현황": "population",
    "가설 검증 분석": "hypothesis",
    "SQL 쿼리 설명": "sql_info",
    "인사이트 & AI 브리핑": "insight"
}
selection = st.sidebar.radio("섹션을 선택하세요", list(sections.keys()))

# --- (A) 헤더 + 핵심 지표 카드 ---
st.markdown(f"<div id='header'></div>", unsafe_allow_html=True)
st.title("결혼이주여성 14만 시대, 왜 10명 중 1명은 죽음을 생각하는가?")
st.markdown("#### 정부의 '동화 중심' 정책과 이주여성의 '심리적 고립' 사이의 미스매치 분석")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("결혼이민자 총수(2024)", "181,436명")
with col2:
    st.metric("여성 결혼이민자(80.3%)", "145,731명")
with col3:
    st.metric("우울장애 유병률", "8.3%", delta="일반인 대비 높음")
with col4:
    st.metric("자살 생각 경험률", "12.9%", delta="심각", delta_color="inverse")

st.divider()

# --- (B) 데이터 설명 ---
if selection == "데이터 개요":
    st.markdown(f"<div id='data_info'></div>", unsafe_allow_html=True)
    st.subheader("📊 데이터 구성 및 출처")
    
    data_desc = [
        {"Table": "mental_health_overall", "내용": "우울/자살생각 유병률", "출처": "보사연/질병청(2024)"},
        {"Table": "mental_health_by_characteristic", "내용": "특성별(차별경험 등) 정신건강", "출처": "보사연/질병청(2024)"},
        {"Table": "suicide_reason", "내용": "자살생각 원인(복수응답)", "출처": "보사연/질병청(2024)"},
        {"Table": "mental_health_care", "내용": "상담 인지 및 이용 경험", "출처": "보사연/질병청(2024)"},
        {"Table": "perception_immigrant", "내용": "한국인의 사회문제 인식", "출처": "KOSSDA(2024)"},
        {"Table": "immigrant_trend_total", "내용": "연도별 총 이민자 수", "출처": "법무부 통계연보"},
        {"Table": "immigrant_by_gender/nationality", "내용": "성별/국적별 분포", "출처": "법무부 통계연보"},
        {"Table": "pop_female_sido", "내용": "시도별 여성 이민자 수", "출처": "행안부(2024)"},
        {"Table": "infra_center/edu_sido", "내용": "지원센터/교육기관 인프라", "출처": "여가부/한가진(2025)"}
    ]
    st.table(pd.DataFrame(data_desc))

# --- (C) 인구통계 시각화 파트 (오류 방지 동적 로드) ---
elif selection == "인구통계 현황":
    st.markdown(f"<div id='population'></div>", unsafe_allow_html=True)
    st.subheader("📊 지역별 인구 및 증가 추세")
    
    col_pop, col_trend = st.columns([1, 1])
    
    with col_pop:
        st.write("##### 지역별 여성 결혼이민자 수")
        df_pop = run_query("SELECT * FROM pop_female_sido")
        
        if not df_pop.empty:
            # 컬럼명이 영어든 한글이든 상관없이 인덱스로 바인딩 (KeyError 원천 차단)
            x_col = 'sido' if 'sido' in df_pop.columns else df_pop.columns[0]
            y_col = 'female_marriage_immigrants' if 'female_marriage_immigrants' in df_pop.columns else df_pop.columns[1]
            
            fig_pop = px.bar(df_pop, x=x_col, y=y_col, text_auto='.2s', color=y_col, color_continuous_scale='Blues')
            fig_pop.update_layout(xaxis_title="지역", yaxis_title="인구 수")
            st.plotly_chart(fig_pop, use_container_width=True)
        else:
            st.error("지역 인구 데이터(pop_female_sido)를 불러올 수 없습니다. DB 파일 혹은 테이블을 확인하세요.")

    with col_trend:
        st.write("##### 연도별 증가 추이")
        df_trend = run_query("SELECT * FROM immigrant_trend_total")
        if not df_trend.empty:
            x_trend = 'year' if 'year' in df_trend.columns else df_trend.columns[0]
            y_trend = 'total_count' if 'total_count' in df_trend.columns else df_trend.columns[1]
            
            fig_trend = px.line(df_trend, x=x_trend, y=y_trend, markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.error("연도별 추이 데이터(immigrant_trend_total)를 불러올 수 없습니다.")

    st.divider()
    
    # 국적별 막대
    st.write("##### 국적별 분포 (2024)")
    df_nat = run_query("SELECT * FROM immigrant_by_nationality")
    if not df_nat.empty:
        # 2024년 데이터가 있으면 필터링, 없으면 전체 출력
        if 'year' in df_nat.columns:
            df_nat = df_nat[df_nat['year'] == 2024] if 2024 in df_nat['year'].values else df_nat
            
        x_nat = 'nationality' if 'nationality' in df_nat.columns else df_nat.columns[min(2, len(df_nat.columns)-1)]
        y_nat = 'n' if 'n' in df_nat.columns else df_nat.columns[min(3, len(df_nat.columns)-1)]
        
        fig_nat = px.bar(df_nat, x=x_nat, y=y_nat, color=y_nat, title="국적별 비중 분석")
        st.plotly_chart(fig_nat, use_container_width=True)
    else:
        st.warning("국적별 데이터가 존재하지 않습니다.")

# --- (D) 가설 관련 시각화 파트 (오류 방지 동적 로드) ---
elif selection == "가설 검증 분석":
    st.markdown(f"<div id='hypothesis'></div>", unsafe_allow_html=True)
    st.subheader("🧪 가설 검증: 지역적 밀집이 차별적 태도를 만드는가?")
    
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**H1: 집단위협 가설**\n\n지역 내 여성 이민자 비율이 높을수록 배타성이 높아지는가?\n\n**결과:** {ANALYSIS_RESULTS['H1']['decision']}\n(계수: {ANALYSIS_RESULTS['H1']['coef_region']}, p < 0.001)")
    with c2:
        st.info(f"**H2: 밀집도 차이**\n\n상위 밀집지역 vs 하위 밀집지역 배타성 차이\n\n**결과:** {ANALYSIS_RESULTS['H2']['decision']}\n(t={ANALYSIS_RESULTS['H2']['t_stat']}, p={ANALYSIS_RESULTS['H2']['p_val']})")

    st.divider()
    
    # 핵심 시각화: 차별경험별 정신건강
    st.subheader("🚨 핵심 요인: 차별 경험이 정신건강에 미치는 영향")
    df_char_all = run_query("SELECT * FROM mental_health_by_characteristic")
    
    if not df_char_all.empty:
        # 컬럼 이름이 달라도 깨지지 않도록 순서 지정
        group_col = df_char_all.columns[0] 
        cat_col = df_char_all.columns[1]   
        sui_col = df_char_all.columns[2]   
        dep_col = df_char_all.columns[3]   

        # '차별' 문자가 포함된 행 필터링 시도
        df_char = df_char_all[df_char_all[group_col].astype(str).str.contains('차별')]
        if df_char.empty: # 필터링 결과가 없으면 전체 데이터를 시각화 대상으로 지정
            df_char = df_char_all
        
        try:
            fig_mental = go.Figure()
            fig_mental.add_trace(go.Bar(x=df_char[cat_col], y=df_char[sui_col], name='자살 생각(%)'))
            fig_mental.add_trace(go.Bar(x=df_char[cat_col], y=df_char[dep_col], name='우울 장애(%)'))
            fig_mental.update_layout(title="차별 경험 유무에 따른 정신건강 지표 위험도 차이", barmode='group')
            st.plotly_chart(fig_mental, use_container_width=True)
        except Exception as e:
            st.error(f"차트 시각화 중 오류가 발생했습니다: {e}")
            st.dataframe(df_char_all.head())
    else:
        st.error("정신건강 데이터(mental_health_by_characteristic)를 불러올 수 없습니다.")

    st.divider()

    # 돌봄 공백 깔때기
    st.subheader("📉 서비스 전달 체계의 누수 (돌봄 공백)")
    funnel_data = dict(
        number=[100, 88.3, 9.8, 3.2], 
        stage=["전체 응답자", "상담기관 인지", "상담 필요성 인지", "실제 상담 이용"]
    )
    fig_funnel = px.funnel(funnel_data, x='number', y='stage', title="도움이 필요해도 닿지 못하는 구조")
    st.plotly_chart(fig_funnel, use_container_width=True)

# --- (E) SQL 설명 ---
elif selection == "SQL 쿼리 설명":
    st.markdown(f"<div id='sql_info'></div>", unsafe_allow_html=True)
    st.subheader("🔍 주요 SQL 분석 로직")
    
    with st.expander("시도별 인구와 인프라 매칭 (미스매치 분석)"):
        st.code("""
SELECT p.sido, p.female_marriage_immigrants AS 여성인구,
       c.center_count AS 지원센터, e.org_count AS 한국어교육기관
FROM pop_female_sido p
LEFT JOIN infra_center_sido c ON p.sido = c.sido
LEFT JOIN infra_korean_edu_sido e ON p.sido = e.sido
ORDER BY 여성인구 DESC;
        """, language='sql')
        st.write("지역별 이주여성 인구 대비 지원 시설의 불균형을 파악하기 위한 조인 쿼리입니다.")

# --- (F) 인사이트 & AI 브리핑 ---
elif selection == "인사이트 & AI 브리핑":
    st.markdown(f"<div id='insight'></div>", unsafe_allow_html=True)
    st.subheader("💡 주요 인사이트 요약")
    
    i1, i2, i3 = st.columns(3)
    i1.success("**1. 차별은 치명적인 균열**\n\n차별 경험은 자살 생각을 5배 증폭시키는 단일 최대 위험 인자입니다.")
    i2.warning("**2. 지역보단 세대/사회 문제**\n\nH1의 설명력은 2.8%에 불과합니다. 배타성은 특정 지역의 밀집 때문이 아니라 우리 사회 전반의 인식 문제입니다.")
    i3.error("**3. 인프라의 동화 편중**\n\n한국어 교육 기관은 충분하나, 심리 상담으로 이어지는 깔때기는 매우 좁습니다(누수 심각).")

    st.divider()
    
    st.subheader("🤖 Gemini AI 정책 브리핑 생성")
    if st.button("AI 브리핑 생성하기"):
        if "GEMINI_API_KEY" not in st.secrets:
            st.warning("Gemini API 키가 설정되지 않았습니다. `.streamlit/secrets.toml`을 확인하세요.")
        else:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-pro')
                prompt = f"""
                당신은 보건복지부 정책 보좌관입니다. 아래 데이터를 바탕으로 결혼이주여성 정신건강 정책 제언을 3-5문장으로 작성하세요.
                - 여성 결혼이민자 14.5만명, 자살생각 12.9%
                - 차별 경험 시 자살생각 5배 증가(30.6%)
                - 인프라는 한국어 교육에 편중, 상담 서비스 이용은 3.2% 수준으로 매우 낮음
                - 가설 검증 결과: 특정 지역 밀집보다는 사회 전체적 배타성이 문제임
                """
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"AI 호출 중 오류가 발생했습니다: {e}")

    st.caption("**한계점:** 본 분석은 시도 단위 매핑을 가정하며, 상관관계가 반드시 인과관계를 의미하지는 않습니다.")