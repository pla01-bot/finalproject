# -*- coding: utf-8 -*-
"""
[경영정보처리론] 분석 스크립트 — 한국 사회의 '시선'과 결혼이주여성 거주 분포
주제: "결혼이주여성 14만 시대, 왜 10명 중 1명은 죽음을 생각하는가?"

[연구 목적]
  정부·지자체는 인구 방어를 위해 결혼이주여성의 '유입'을 장려해 왔다.
  그렇다면 정작 이들을 둘러싼 한국 사회의 '시선(이주민에 대한 배타적 태도)'은
  결혼이주여성이 많이 사는 지역에서 어떻게 나타나는가? 이를 개인 단위 원자료로 검증한다.

[가설]
  H1 (집단위협가설): 결혼이주여성이 많이 거주하는 지역일수록 한국인의 배타적 태도가 높다(정적).
      ↔ 대립(접촉가설): 많이 접할수록 수용적이다(부적).
  H2: 결혼이주여성 밀집 '상위' 지역과 '하위' 지역 간 배타적 태도 평균에 차이가 있다.

[데이터/방법]
  - 개인: 2021 국민 다문화수용성 조사 (n=5,000)
  - 지역: 시도별 여성 결혼이민자 수 (pop_female_sido) — SQL JOIN으로 결합
  - 분석: 다중회귀(OLS) + 독립표본 t-test, 척도 신뢰도(Cronbach's α)
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

SRC = "/sessions/wizardly-inspiring-wright/mnt/Data/2021년_국민 다문화수용성 조사 데이터.csv"
DB = "/tmp/multicultural.db"   # build_database.py가 만든 DB(로컬). 없으면 cleaned에서 적재.
CLEAN = "/sessions/wizardly-inspiring-wright/mnt/경영정보처리론/cleaned"

# 지역구분 코드(1~17) → 시도 약칭. (통계청 표준 시도 배열 가정 — 코드북 확인 권장)
SIDO_CODE = {1: "서울", 2: "부산", 3: "대구", 4: "인천", 5: "광주", 6: "대전",
             7: "울산", 8: "세종", 9: "경기", 10: "강원", 11: "충북", 12: "충남",
             13: "전북", 14: "전남", 15: "경북", 16: "경남", 17: "제주"}


def read_csv_any(path):
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise IOError(path)


def cronbach_alpha(items_df):
    """척도 신뢰도 Cronbach's α 계산."""
    items_df = items_df.dropna()
    k = items_df.shape[1]
    var_sum = items_df.var(axis=0, ddof=1).sum()
    total_var = items_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_sum / total_var)


def main():
    df = read_csv_any(SRC)

    # ---- 1) 배타적 태도 지수 생성 (다양성2 + 다양성3 + 관계성1, 15문항) ----
    cols = list(df.columns)
    item_cols = [c for c in cols if c.startswith("다양성차원 척도 2")
                 or c.startswith("다양성차원 척도 3")
                 or c.startswith("관계성차원 척도 1")]
    # 1~6점 외(예: 9=모름)은 결측 처리
    items = df[item_cols].apply(lambda s: s.where(s.between(1, 6)))
    alpha = cronbach_alpha(items)
    df["exclusion_score"] = items.mean(axis=1)   # 높을수록 배타적(차가운 시선)

    # ---- 2) 인구통계/지역 변수 정리 ----
    df["age"] = df["연령대"]
    df["female"] = (df["성별"] == 2).astype(int)         # 1=여성
    df["media_contact"] = (df["최근 1년간 대중매체 외국 이주민 접촉 경험 유무"] == 1).astype(int)
    df["region_size"] = df["지역규모"]                   # 1~3
    df["sido"] = df["지역구분"].map(SIDO_CODE)

    base = df[["sido", "exclusion_score", "age", "female",
               "media_contact", "region_size"]].dropna()

    # ---- 3) SQL JOIN: 개인 ⋈ 시도별 여성 결혼이민자 수 ----
    con = sqlite3.connect(":memory:")
    base.to_sql("acceptance", con, index=False)
    pop = read_csv_any(f"{CLEAN}/marriage_immigrant_pop_by_sido.csv")
    pop.to_sql("pop_female_sido", con, index=False)

    joined = pd.read_sql_query("""
        SELECT a.*, p.female_marriage_immigrants AS region_women
        FROM acceptance a
        JOIN pop_female_sido p ON a.sido = p.sido      -- 공통 키: 시도
    """, con)
    con.close()
    # 인구 규모는 스케일이 커서 로그·만명 단위로 변환(해석 편의)
    joined["region_women_10k"] = joined["region_women"] / 10000.0

    print("=" * 70)
    print(f"분석 표본 n = {len(joined):,}")
    print(f"배타적 태도 지수: 15문항, Cronbach's α = {alpha:.3f} "
          f"(평균 {joined['exclusion_score'].mean():.2f} / 6점)")

    # ---- 4) H1 다중회귀 (OLS) ----
    print("\n" + "=" * 70)
    print("[H1] 다중회귀: 배타적 태도 ~ 지역 여성결혼이민자수 + 통제변수")
    model = smf.ols(
        "exclusion_score ~ region_women_10k + age + female + media_contact + C(region_size)",
        data=joined).fit()
    print(model.summary().tables[1])
    b = model.params["region_women_10k"]
    p = model.pvalues["region_women_10k"]
    print(f"\n핵심: 지역 여성결혼이민자 1만명↑ 당 배타적태도 {b:+.4f}점 변화 (p={p:.4f})")

    # ---- 5) H2 독립표본 t-test (밀집 상위 vs 하위 지역) ----
    print("\n" + "=" * 70)
    print("[H2] t-test: 결혼이주여성 밀집 상위 지역 vs 하위 지역")
    med = joined["region_women"].median()
    high = joined[joined["region_women"] > med]["exclusion_score"]
    low = joined[joined["region_women"] <= med]["exclusion_score"]
    t, pt = stats.ttest_ind(high, low, equal_var=False)
    print(f"  상위(밀집) 지역 n={len(high):,}, 평균={high.mean():.3f}")
    print(f"  하위 지역      n={len(low):,}, 평균={low.mean():.3f}")
    print(f"  t = {t:.3f}, p = {pt:.4f}")

    # 결과 저장
    joined.to_csv(f"{CLEAN}/analysis_dataset.csv", index=False, encoding="utf-8-sig")
    print("\n분석 데이터셋 저장:", f"{CLEAN}/analysis_dataset.csv")


if __name__ == "__main__":
    main()
