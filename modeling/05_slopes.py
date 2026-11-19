# 5단계는 종별 기온 기울기(1도당 일수)와 가정 입력에 쓰는 전체 기울기를 낸다
#
# 기울기는 조합마다 평균을 뺀 날짜를 조합마다 평균을 뺀 기온에 맞춘 값이다
# M1과 같은 식이고 전체 해를 쓴다
# 가정 입력은 기능 1과 2의 기온 가정 인자다
# 여기에는 상한 줄 입력(기온 2~3월 변수)의 전체 기울기를 쓴다(6절 1항)
# 표준오차는 조합 수만큼 자유도를 뺀 잔차로 낸다
# 종별 값은 해 수가 적어 참고로만 본다

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

import mconfig as C
import models as M

COLS = ["현상", "목표단계", "종", "종집단", "입력", "자료판", "조합수", "줄수", "기울기_일per도", "표준오차", "하한95",
        "상한95", "상관계수", "쓰임"]


def slope(g, x):
    t = g.dropna(subset=[x, "y"])
    if t["조합"].nunique() == 0 or len(t) < 3:
        return None
    xx = t[x] - t.groupby("조합")[x].transform("mean")
    yy = t["y"] - t.groupby("조합")["y"].transform("mean")
    sxx = float((xx ** 2).sum())
    if sxx == 0:
        return None
    b = float((xx * yy).sum() / sxx)
    df = len(t) - t["조합"].nunique() - 1
    se = float(np.sqrt(((yy - b * xx) ** 2).sum() / df / sxx)) if df > 0 else np.nan
    q = stats.t.ppf(0.975, df) if df > 0 else np.nan
    return {"조합수": t["조합"].nunique(), "줄수": len(t), "기울기_일per도": b, "표준오차": se,
            "하한95": b - q * se, "상한95": b + q * se, "상관계수": float(np.corrcoef(xx, yy)[0, 1])}


def main():
    mode = C.mode()
    fdir = C.out_dir("features")
    main_lab, other_lab = ("대구 보정", "대구 원값") if C.DAEGU_ADJUST else ("대구 원값", "대구 보정")
    other_file = C.FEATURES_RAW.name if C.DAEGU_ADJUST else "weather_features_daegu_adj.csv"
    st = C.STAGE_SPRING if mode == "정제본" else "수목원 연도 효과(집계표)"
    rows = []
    for lab, path in ((main_lab, fdir / C.FEATURES.name), (other_lab, fdir / other_file)):
        Z = M.load(path)
        Z = Z[(Z["현상"] == "개화") & (Z["목표단계"] == st)]
        parts = [("전체", "전체", Z)]
        if lab == main_lab and mode == "정제본":
            parts += [(g, g, h) for g, h in Z.groupby("종집단")]
            parts += [(sp, h["종집단"].iloc[0], h) for sp, h in Z.groupby("종")]
        for name, grp, g in parts:
            for x in (C.M1_MAIN, C.M1_UPPER):
                r = slope(g, x)
                if r is None:
                    continue
                use = "가정 입력 기울기" if (name == "전체" and x == C.M1_UPPER and lab == main_lab) else ""
                rows.append({"현상": "개화", "목표단계": st, "종": name, "종집단": grp, "입력": x, "자료판": lab, **r,
                             "쓰임": use})
    T = pd.DataFrame(rows).reindex(columns=COLS)
    for c in ("기울기_일per도", "표준오차", "하한95", "상한95", "상관계수"):
        T[c] = T[c].round(2)
    path = C.out_dir("tables") / "model_species_slope.csv"
    T.to_csv(path, index=False, encoding="utf-8-sig")
    print("자료판:", mode, "| 저장:", path.relative_to(C.ROOT), T.shape)
    print(T[T["종"].isin(["전체", "이른 종", "늦은 종"])].to_string(index=False))
    if mode == "정제본":
        sp = T[(T["입력"] == C.M1_UPPER) & ~T["종"].isin(["전체", "이른 종", "늦은 종"])]
        print(sp[["종", "종집단", "조합수", "줄수", "기울기_일per도", "하한95", "상한95"]].sort_values("기울기_일per도").to_string(index=False))
    s = T[T["쓰임"] == "가정 입력 기울기"].iloc[0]
    print(f"가정 입력 기울기: {s['기울기_일per도']}일/도 (95% {s['하한95']}~{s['상한95']}, {s['줄수']}줄). 임시 구현의 -3.56과 견줄 것")


if __name__ == "__main__":
    main()
