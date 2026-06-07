# -*- coding: utf-8 -*-
"""
[경영정보처리론 기말 프로젝트] 1단계 전처리 스크립트
주제: "결혼이주여성 14만 시대, 왜 10명 중 1명은 죽음을 생각하는가?"

처리 내용
  STEP 1) 정신건강 4개 테이블 통일 (병합셀 ffill → 공통 long 스키마)
  STEP 2) KOSSDA(한국인이 바라본 사회문제 2024) 변수표 가공 (보기별 비율 계산)
  STEP 3) 시도명 표준화 + 지역 데이터 시도 단위 집계 (JOIN 키 정합성 확보)

원칙
  - 원자료 수치를 그대로 보존한다. 연구에 유리하도록 값을 만들어내지 않는다.
  - 모든 산출 컬럼명은 SQL 친화형(영문 snake_case, 공백/특수문자 제거)으로 통일한다.
  - 값(범주명)은 한국어 원문을 유지해 데이터 정합성을 지킨다.
"""

import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 입력(원본 데이터) / 출력(정제 데이터) 경로
SRC = "/sessions/wizardly-inspiring-wright/mnt/Data"
OUT = "/sessions/wizardly-inspiring-wright/mnt/경영정보처리론/cleaned"
os.makedirs(OUT, exist_ok=True)


def read_csv_any(path):
    """한글 인코딩이 섞여 있어 여러 인코딩을 순차 시도하는 CSV 리더."""
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise IOError(f"읽기 실패: {path}")


# ======================================================================
# STEP 1) 정신건강 4개 테이블 통일
# ----------------------------------------------------------------------
# 4개 엑셀은 모두 '보건사회연구원·질병관리청 2024년 결혼이주여성 519명 조사'에서
# 나온 집계표이다. 엑셀의 첫 컬럼(구분)은 병합셀이라 NaN이 섞여 있으므로
# ffill(앞 값으로 채우기)로 복원한 뒤 공통 스키마로 정리한다.
# ======================================================================

def step1_mental_health():
    # --- (1-1) 전체 정신건강 유병 현황 -------------------------------
    # 원자료: 구분 / 세부구분 / 사례수(N) / 비율(%)
    df = pd.read_excel(f"{SRC}/결혼이주여성의 정신건강_2024년.xlsx", header=0)
    df.columns = ["indicator_kr", "category", "n", "pct"]
    df["indicator_kr"] = df["indicator_kr"].ffill()  # 병합셀 복원
    # 한글 지표명을 영문 코드로 매핑 (SQL/시각화 편의)
    ind_map = {"우울장애 유병": "depression",
               "범불안장애 경험": "anxiety",
               "자살생각 여부": "suicidal_ideation"}
    df["indicator"] = df["indicator_kr"].map(ind_map)
    mh_overall = df[["indicator", "indicator_kr", "category", "n", "pct"]]
    mh_overall.to_csv(f"{OUT}/mental_health_overall.csv", index=False, encoding="utf-8-sig")

    # --- (1-2) 특성별 정신건강 수준 (★핵심: 차별경험 등) ------------
    # 원자료: 구분 / 세부특성 / 우울장애 유병(%) / 우울장애 유병(N)
    #         / 자살생각(%) / 자살생각(N) / 전체 사례 수
    df = pd.read_excel(f"{SRC}/결혼이주여성의 특성별 정신건강 수준_2024년.xlsx", header=0)
    df.columns = ["char_group", "char_category",
                  "depression_pct", "depression_n",
                  "suicide_pct", "suicide_n", "total_n"]
    df["char_group"] = df["char_group"].ffill()  # 병합셀(구분) 복원
    mh_char = df.dropna(subset=["char_category"]).reset_index(drop=True)
    mh_char.to_csv(f"{OUT}/mental_health_by_characteristic.csv", index=False, encoding="utf-8-sig")

    # --- (1-3) 자살생각 원인 (1·2순위 복수응답) ---------------------
    df = pd.read_excel(f"{SRC}/결혼이주여성의 자살생각 원인_2024년.xlsx", header=0)
    df.columns = ["reason", "n", "pct"]
    df.to_csv(f"{OUT}/suicide_reason.csv", index=False, encoding="utf-8-sig")

    # --- (1-4) 정신건강 인지 및 상담 경험 (돌봄 공백) ---------------
    # 주의: '상담 경험' 항목의 분모(N=51)는 '상담 필요 인지자'로 다른 항목과
    #       모집단이 다르다. 비율은 원자료 그대로 보존한다.
    df = pd.read_excel(f"{SRC}/결혼이주여성의 정신건강 문제에 대한 인지 및 상담 경험_2024년.xlsx", header=0)
    df.columns = ["care_item", "response", "n", "pct"]
    df["care_item"] = df["care_item"].ffill()  # 병합셀 복원
    df.to_csv(f"{OUT}/mental_health_care.csv", index=False, encoding="utf-8-sig")

    return mh_overall, mh_char


# ======================================================================
# STEP 2) KOSSDA 변수표 가공 (필수 데이터)
# ----------------------------------------------------------------------
# 한국인 1,000명 대상 '외국인 이주민' 인식 변수. 변수명(Label)이 병합셀이라
# ffill로 복원하고, 변수별 합계 대비 보기(Category)별 비율을 계산한다.
# 모든 보기를 보존하여 자의적 취사선택(데이터 만들어내기)을 방지한다.
# ======================================================================

def step2_kossda():
    df = pd.read_excel(
        f"{SRC}/한국인이 바라본 사회문제 2024_외국인 이주민 변수 데이터.xlsx",
        sheet_name="Variables", header=0)
    df = df[["Label", "Category value", "Category label", "Category stat"]].copy()
    df.columns = ["variable_kr", "category_value", "category_label", "n"]
    df["variable_kr"] = df["variable_kr"].ffill()  # 변수명 병합셀 복원
    df = df.dropna(subset=["n"])
    df["n"] = df["n"].astype(int)

    # 변수별 응답 합계를 분모로 비율(%) 계산 — 원자료 빈도만 사용
    df["total_n"] = df.groupby("variable_kr")["n"].transform("sum")
    df["pct"] = (df["n"] / df["total_n"] * 100).round(1)

    perception = df[["variable_kr", "category_value", "category_label", "n", "total_n", "pct"]]
    perception.to_csv(f"{OUT}/perception_immigrant.csv", index=False, encoding="utf-8-sig")
    return perception


# ======================================================================
# STEP 3) 시도명 표준화 + 지역 데이터 집계 (JOIN 키 정합성)
# ----------------------------------------------------------------------
# 파일마다 시도 표기가 다르다.
#   인구파일 : '서울특별시','강원특별자치도' (정식명칭)
#   지원센터 : '부산광역시','경기도'         (정식명칭)
#   교육기관 : '서울','강원','경기'          (약칭)
# 모두 약칭(17개 시도)으로 통일해 JOIN 키를 일치시킨다. (협업규칙 4번)
# ======================================================================

# 정식명칭/변형 → 표준 약칭 매핑 딕셔너리
SIDO_MAP = {
    "서울특별시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산": "부산",
    "대구광역시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천": "인천",
    "광주광역시": "광주", "광주": "광주",
    "대전광역시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종특별시": "세종", "세종": "세종",
    "경기도": "경기", "경기": "경기",
    "강원특별자치도": "강원", "강원도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남",
    "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}


def norm_sido(name):
    """시도명을 표준 약칭으로 변환. 매핑에 없으면 원본 유지(누락 추적용)."""
    if pd.isna(name):
        return name
    return SIDO_MAP.get(str(name).strip(), str(name).strip())


def step3_region():
    # --- (3-1) 결혼이주여성(여) 인구: 시군구→시도, 체류기간 '합계'만 ----
    df = read_csv_any(f"{SRC}/시군구별_체류기간별_외국인_주민현황_여__20260527142755.csv")
    # 0번째 행은 한글 헤더(설명행)이므로 컬럼 재지정 후 제거
    df.columns = ["sido_raw", "stay_period", "total", "foreign_worker",
                  "marriage_immigrant", "student", "overseas_korean", "etc_foreign"]
    df = df[df["stay_period"] == "합계"]            # 체류기간 전체 합계 행만
    df = df[df["sido_raw"] != "합계"]               # 전국 합계 행 제외
    df["sido"] = df["sido_raw"].apply(norm_sido)    # 시도명 표준화
    pop = df[["sido", "marriage_immigrant"]].copy()
    pop["marriage_immigrant"] = pd.to_numeric(pop["marriage_immigrant"], errors="coerce").astype("Int64")
    pop = pop.rename(columns={"marriage_immigrant": "female_marriage_immigrants"})
    pop = pop.groupby("sido", as_index=False)["female_marriage_immigrants"].sum()
    pop.to_csv(f"{OUT}/marriage_immigrant_pop_by_sido.csv", index=False, encoding="utf-8-sig")

    # --- (3-2) 다문화가족지원센터: 시도별 센터 수·직원 수 -------------
    df = read_csv_any(f"{SRC}/한국건강가정진흥원_다문화가족지원센터 현황_20250924.csv")
    df["sido"] = df["시도"].apply(norm_sido)
    df["직원수"] = pd.to_numeric(df["직원수"], errors="coerce")
    center = df.groupby("sido").agg(
        center_count=("센터", "count"),
        staff_total=("직원수", "sum")).reset_index()
    center.to_csv(f"{OUT}/multicultural_center_by_sido.csv", index=False, encoding="utf-8-sig")

    # --- (3-3) 한국어교육 운영기관: 시도별 기관 수 -------------------
    df = read_csv_any(f"{SRC}/성평등가족부_결혼이민자 대상 한국어교육 운영기관 현황_20251231.csv")
    df["sido"] = df["시도"].apply(norm_sido)
    edu = df.groupby("sido").agg(org_count=("운영기관명", "count")).reset_index()
    edu.to_csv(f"{OUT}/korean_edu_org_by_sido.csv", index=False, encoding="utf-8-sig")

    return pop, center, edu


if __name__ == "__main__":
    print("STEP 1) 정신건강 4개 테이블 통일 ...")
    mh_overall, mh_char = step1_mental_health()
    print("STEP 2) KOSSDA 변수표 가공 ...")
    perception = step2_kossda()
    print("STEP 3) 시도명 표준화 + 지역 집계 ...")
    pop, center, edu = step3_region()

    # ---- 검증 리포트 ----
    print("\n" + "=" * 60)
    print("[검증] 정신건강 전체:", mh_overall.shape, "| 특성별:", mh_char.shape)
    print("[검증] KOSSDA 변수 수:", perception["variable_kr"].nunique(),
          "| 행:", perception.shape[0])
    print("[검증] 인구 시도:", sorted(pop["sido"]))
    print("[검증] 센터 시도:", sorted(center["sido"]))
    print("[검증] 교육 시도:", sorted(edu["sido"]))
    j = pop.merge(center, on="sido", how="left").merge(edu, on="sido", how="left")
    print("\n[JOIN 미리보기] 시도 기준 LEFT JOIN")
    print(j.to_string(index=False))
    print("\n완료. 출력 폴더:", OUT)
