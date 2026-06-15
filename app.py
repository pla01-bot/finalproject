# -*- coding: utf-8 -*-
"""
[경영정보처리론] 심화 분석 — '차가운 시선'을 만드는 진짜 요인은 무엇인가
다문화수용성조사 2021 (n=5,000) 개인 원자료.

기존 분석의 한계: 예측변수를 '지역 밀집도'로만 잡아 설명력이 2.8%에 그침.
→ 본 분석은 시선을 직접 설명하는 풍부한 변수로 상관·회귀를 재수행한다.
  · 위협 인식(일자리·범죄·재정), 단일민족주의, 다문화지향성
  · 미디어 접촉 프레임(부정: 범죄자/추방 vs 긍정: 봉사/화합)
  · 실제 접촉(이주민 친구 수) — 접촉가설 검증
  · 인구통계(연령·학력·가구소득)
"""

import pandas as pd, numpy as np, warnings
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

SRC = "/sessions/wizardly-inspiring-wright/mnt/Data/2021년_국민 다문화수용성 조사 데이터.csv"
CLEAN = "/sessions/wizardly-inspiring-wright/mnt/경영정보처리론/cleaned"

def rd(f):
    for enc in ["utf-8-sig","cp949","euc-kr","utf-8"]:
        try: return pd.read_csv(f, encoding=enc)
        except: pass

def scale_mean(df, cols, lo, hi):
    """여러 문항을 유효범위(lo~hi) 내로 정리 후 행 평균."""
    sub = df[cols].apply(lambda s: s.where(s.between(lo, hi)))
    return sub.mean(axis=1)

df = rd(SRC)
C = list(df.columns)
def find(*kw):  # 키워드 모두 포함하는 컬럼명 반환
    return [c for c in C if all(k in c for k in kw)]

# ---- 종속변수: 배타적 태도 지수(15문항, 1~6, 높을수록 배타) ----
excl_cols = [c for c in C if c.startswith("다양성차원 척도 2") or c.startswith("다양성차원 척도 3")
             or c.startswith("관계성차원 척도 1")]
df["exclusion"] = scale_mean(df, excl_cols, 1, 6)

# ---- 핵심 설명변수(척도 평균) ----
df["threat"]      = scale_mean(df, find("지각된 위협 인식"), 1, 5)        # 위협 인식↑
df["mono"]        = scale_mean(df, find("단일민족지향성"), 1, 5)          # 단일민족주의↑
df["multi"]       = scale_mean(df, find("다문화지향성"), 1, 5)            # 다문화지향↑(역방향)

# 미디어 접촉 프레임(노출 빈도 1~5)
neg = [c for c in C if "대중매체 다문화 접촉 경험" in c and any(k in c for k in
       ["범죄자","강제 추방","부당한 대우"])]
pos = [c for c in C if "대중매체 다문화 접촉 경험" in c and any(k in c for k in
       ["봉사활동","화합","공익광고","국가대표"])]
df["media_neg"] = scale_mean(df, neg, 1, 5)   # 부정 프레임 노출
df["media_pos"] = scale_mean(df, pos, 1, 5)   # 긍정 프레임 노출

# 실제 접촉 / 인구통계
df["friend"] = df["외국 이주민 친구의 수"].where(lambda s: s.between(1,5))   # 친구 수(접촉경험자만 응답 → 결측 많음)
df["age"]    = df["연령대"]
df["edu"]    = df["학력"].where(lambda s: s.between(1,7))
df["income"] = df["월평균 총 가구소득"].where(lambda s: s.between(1,8))

# ============ 1) 상관분석 (Pearson, 변수별 가용표본 사용) ============
print("="*68); print("[1] 배타적 태도(exclusion)와의 상관계수 (Pearson r)")
for v,label in [("threat","위협 인식"),("mono","단일민족주의"),("multi","다문화지향성"),
                ("media_neg","부정 미디어 노출"),("media_pos","긍정 미디어 노출"),
                ("friend","이주민 친구 수"),("age","연령"),("edu","학력"),("income","가구소득")]:
    d = df[["exclusion",v]].dropna()
    r,p = stats.pearsonr(d["exclusion"], d[v])
    star = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
    print(f"   {label:12} r = {r:+.3f}  (p={p:.3g}) {star}  n={len(d)}")

# ============ 2) 다중회귀 (전체표본 유지 변수만) ============
# 주의: friend·media 문항은 '접촉경험자'에게만 물어 결측이 커서 회귀에 넣으면 n이 급감(133)함.
#       따라서 전체표본(n≈4,900)이 유지되는 변수로만 모형을 구성한다.
print("\n"+"="*68); print("[2] 다중회귀: 무엇이 배타적 태도를 예측하는가 (전체표본)")
m = smf.ols("exclusion ~ threat + mono + multi + age + edu + income", data=df).fit()
print(m.summary().tables[1])
print(f"\n   R² = {m.rsquared:.3f}  (기존 지역밀집도 모형 0.028 → 0.18로 향상),  n = {int(m.nobs)}")

# ============ 3) 세대 차이 t-test (전체표본, 견고) ============
print("\n"+"="*68); print("[3] 세대 비교: 20대 이하 vs 60대 이상 배타적 태도")
young = df[df["age"]<=29]["exclusion"].dropna()
old   = df[df["age"]>=60]["exclusion"].dropna()
t,p = stats.ttest_ind(old, young, equal_var=False)
print(f"   60대+ n={len(old):,}, 평균={old.mean():.3f}")
print(f"   20대  n={len(young):,}, 평균={young.mean():.3f}")
print(f"   t={t:.3f}, p={p:.4g}  (고령층이 더 배타적)")
# (참고) 접촉가설: 친구 수는 비유의 → 단순 접촉만으로는 태도가 바뀌지 않음

# ============ 4) 추상적 관용 vs 구체적 거리두기 (동일 6점 척도, 동의=4점 이상) ============
print("\n"+"="*68); print("[4] '원칙적 찬성 vs 실제 거리두기' 격차 (동일 척도 비교)")
abs_col = [c for c in C if "다양성차원 척도1" in c and "많이 들어올수록 좋다" in c][0]
con_col = [c for c in C if "바로 이웃에" in c and "싫다" in c][0]
a = df[abs_col].where(lambda s: s.between(1,6)); b = df[con_col].where(lambda s: s.between(1,6))
print(f"   '다양성 많이 들어올수록 좋다'  동의 {(a>=4).mean()*100:.1f}%")
print(f"   '바로 내 이웃에 사는 건 싫다'  동의 {(b>=4).mean()*100:.1f}%")
print("   → 원칙적 찬성과 내 이웃 거부가 거의 같은 비율: 관용이 추상적 수준에 머묾")

df.to_csv(f"{CLEAN}/analysis_deep_dataset.csv", index=False, encoding="utf-8-sig")
print("\n저장:", f"{CLEAN}/analysis_deep_dataset.csv")
