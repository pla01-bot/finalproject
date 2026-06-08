# -*- coding: utf-8 -*-
"""
[경영정보처리론] 결혼이민자 현황(2015~2024) 전처리 스크립트
원본: 법무부 출입국·외국인정책 통계연보 - 결혼이민자 현황(연도별·성별·지역별·국적별)

산출(독립 도입 데이터, JOIN 키 없음): 시계열 / 성별 / 국적
  - immigrant_trend_total        : 연도별 전체 결혼이민자 수
  - immigrant_by_gender          : 연도별 성별(남/여) 결혼이민자 수
  - immigrant_by_nationality     : 연도별 국적별 결혼이민자 수

원칙: 원자료 값 그대로 보존. 컬럼명은 SQL 친화형(영문 snake_case).
"""

import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

SRC = "/sessions/wizardly-inspiring-wright/mnt/Data/결혼이민자 현황(연도별·성별·지역별·국적별).xlsx"
OUT = "/sessions/wizardly-inspiring-wright/mnt/경영정보처리론/cleaned"
os.makedirs(OUT, exist_ok=True)


def to_int(x):
    """'145,731' 같은 천단위 콤마 문자열을 정수로 변환."""
    return int(str(x).replace(",", "").strip())


def run():
    # 헤더 없이 통째로 읽는다(표 구조가 불규칙하므로 위치로 접근)
    df = pd.read_excel(SRC, header=None)

    # 2번 행(0-index)에 연도 헤더가 cols 2~11 에 위치 (2015~2024)
    years = [int(df.iloc[2, c]) for c in range(2, 12)]

    # --- (1) 연도별 전체 합계: 3번 행 ---
    total_vals = [to_int(df.iloc[3, c]) for c in range(2, 12)]
    trend = pd.DataFrame({"year": years, "total_count": total_vals})
    trend.to_csv(f"{OUT}/immigrant_trend_total.csv", index=False, encoding="utf-8-sig")

    # --- (2) 성별: 4번(남자), 5번(여자) 행 → long format ---
    gender_rows = {"남자": 4, "여자": 5}
    g_records = []
    for gender, r in gender_rows.items():
        for i, c in enumerate(range(2, 12)):
            g_records.append({"year": years[i], "gender": gender,
                              "n": to_int(df.iloc[r, c])})
    gender = pd.DataFrame(g_records)
    gender.to_csv(f"{OUT}/immigrant_by_gender.csv", index=False, encoding="utf-8-sig")

    # --- (3) 국적: 11~15번 행(중국/베트남/일본/필리핀/기타) → long format ---
    nat_rows = {"중국": 11, "베트남": 12, "일본": 13, "필리핀": 14, "기타": 15}
    n_records = []
    for nat, r in nat_rows.items():
        for i, c in enumerate(range(2, 12)):
            n_records.append({"year": years[i], "nationality": nat,
                              "n": to_int(df.iloc[r, c])})
    nationality = pd.DataFrame(n_records)
    nationality.to_csv(f"{OUT}/immigrant_by_nationality.csv", index=False, encoding="utf-8-sig")

    # ---- 검증 리포트 ----
    print("[검증] 연도 범위:", years[0], "~", years[-1])
    print("[검증] 2024 전체:", trend.loc[trend.year == 2024, "total_count"].iloc[0])
    w = gender[(gender.year == 2024) & (gender.gender == "여자")]["n"].iloc[0]
    t = trend.loc[trend.year == 2024, "total_count"].iloc[0]
    print(f"[검증] 2024 여성: {w} (여성비율 {w/t*100:.1f}%)")
    print("[검증] 2024 국적 상위:\n",
          nationality[nationality.year == 2024].sort_values("n", ascending=False).head(3).to_string(index=False))
    return trend, gender, nationality


if __name__ == "__main__":
    run()
    print("\n완료. 출력 폴더:", OUT)
