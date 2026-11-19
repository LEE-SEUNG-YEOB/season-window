# 6단계 기능 함수가 읽는 서비스 표 8개 저장

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import mconfig as C
import models as M

SPRING_STAGES = [C.STAGE_START, C.STAGE_SPRING, C.STAGE_END]
KEY = ["수목원", "종", "연도"]

def history(P):
    sp = P[P["종"].isin(C.SPRING_SPECIES) & P["현상"].isin(SPRING_STAGES)]
    au = P[P["종"].isin(C.AUTUMN_SPECIES) & (P["현상"] == C.STAGE_AUTUMN)]
    H = pd.concat([sp, au]).dropna(subset=["날짜숫자"])
    if H.duplicated(KEY + ["현상"]).any():
        sys.exit("!! 같은 수목원 종 연도 현상 줄이 겹친다. 정제본을 확인할 것")
    return H[["수목원", "종", "현상", "연도", "날짜숫자", "이상치의심"]].sort_values(["수목원", "종", "현상", "연도"])

def combos(H):
    S = H.groupby(["수목원", "종", "현상"])["날짜숫자"].agg(
        관측연수="count", 평균="mean", 표준편차="std", 가장이른="min", 가장늦은="max").reset_index()
    return S.round({"평균": 3, "표준편차": 3})

def pivot(H, stages):
    W = H[H["현상"].isin(stages)].pivot(index=KEY, columns="현상", values="날짜숫자")
    return W.reindex(columns=stages).dropna().reset_index()

def pairs(H):
    W = pivot(H, [C.STAGE_START, C.STAGE_END]).rename(columns={C.STAGE_START: "개화시작", C.STAGE_END: "낙화"})
    W["개화기간"] = W["낙화"] - W["개화시작"]
    return W[W["개화기간"] > 0][KEY + ["개화시작", "낙화", "개화기간"]]

def gaps(H, B):
    W = pivot(H, [C.STAGE_START, C.STAGE_SPRING])
    W["간격"] = W[C.STAGE_SPRING] - W[C.STAGE_START]
    parts = []
    for kind, by in (("조합", ["수목원", "종"]), ("종", ["종"])):
        a = W.groupby(by)["간격"].agg(간격_중앙값="median", 간격_해수="count")
        b = B.groupby(by)["개화기간"].agg(기간_중앙값="median", 기간_해수="count")
        t = a.join(b, how="outer").reset_index()
        t.insert(0, "구분", kind)
        parts.append(t)
    G = pd.concat(parts, ignore_index=True)
    G["수목원"] = G["수목원"].fillna("")
    for c in ("간격_해수", "기간_해수"):
        G[c] = G[c].fillna(0).astype(int)
    return G[["구분", "수목원", "종", "간격_중앙값", "간격_해수", "기간_중앙값", "기간_해수"]]

def shift(Z):
    rows = []
    for (phen, st), g in Z.groupby(["현상", "목표단계"]):
        last = g["연도"].dropna().drop_duplicates().sort_values().tail(C.RECENT_N)
        rows.append({"현상": phen, "목표단계": st, "시작연도": int(last.min()), "끝연도": int(last.max()),
                     "최근5년_편차": round(M.drift(g), 2), "조합수": g["조합"].nunique(), "줄수": len(g)})
    return pd.DataFrame(rows)

def decision():
    tdir = C.folder("tables")
    D = pd.read_csv(tdir / "model_decision.csv")
    S = pd.read_csv(tdir / "model_species_slope.csv")
    rows = [{"항목": str(r["항목"]).split("(")[0], "현상": r["현상"], "판정": r["판정"], "값": None, "하한95": None,
             "상한95": None, "입력": None, "비고": r.get("기능에서_쓰는_값")} for _, r in D.iterrows()]
    s = S[S["쓰임"] == "가정 입력 기울기"].iloc[0]
    rows.append({"항목": "가정 입력 기울기", "현상": "개화", "판정": "", "값": s["기울기_일per도"], "하한95": s["하한95"],
                 "상한95": s["상한95"], "입력": s["입력"], "비고": f"{s['자료판']}, 조합 {s['조합수']}개 {s['줄수']}줄"})
    return pd.DataFrame(rows)

def lines(Z, features_path):
    t = Z[(Z["현상"] == "개화") & (Z["목표단계"] == C.STAGE_SPRING)]
    b, mx, sd = M.line(t, C.M1_MAIN)
    g = t.groupby("조합")
    T = pd.DataFrame({"평균": g["y"].mean(), "해수": g["y"].count(), "평년_기온": mx, "잔차_표준편차": sd}).reset_index()
    T[["수목원", "종"]] = T["조합"].str.split("|", expand=True)
    T["입력"], T["기울기"] = C.M1_MAIN, round(b, 3)
    T = T[["수목원", "종", "입력", "기울기", "평균", "평년_기온", "잔차_표준편차", "해수"]].round(3)
    X = pd.read_csv(features_path)[["수목원", "연도", C.M1_MAIN]]
    return T, X

# A의 집계표와 비교
# 다르면 정제본 판이 다르거나 정의가 어긋난 것
def parity(S, B):
    msgs = []
    A = pd.read_csv(C.TABLES / "variability_site_species.csv")
    A = A[A["종"].isin(C.SPRING_SPECIES + C.AUTUMN_SPECIES)]
    J = S.merge(A, on=["수목원", "종", "현상"], how="inner", suffixes=("", "_A"))
    bad_n = int((J["관측연수"] != J["관측연수_A"]).sum())
    bad_m = int(((J["평균"] - J["평균_A"]).abs() >= 0.01).sum())
    bad_s = int(((J["표준편차"] - J["표준편차_A"]).abs() >= 0.01).sum())
    msgs.append(f"조합 통계: 견준 조합 {len(J)}개, 관측연수 다름 {bad_n}, 평균 다름 {bad_m}, 표준편차 다름 {bad_s}")
    Hh = pd.read_csv(C.TABLES / "eda_fixed_day_hit.csv")
    mine = B.groupby(["수목원", "종"])["개화기간"].agg(관측연수="count", 개화기간_중앙값="median").reset_index()
    mine = mine[mine["관측연수"] >= C.MIN_YEARS]
    K = mine.merge(Hh, on=["수목원", "종"], how="outer", suffixes=("", "_A"), indicator=True)
    only = int((K["_merge"] != "both").sum())
    both = K[K["_merge"] == "both"]
    bad_k = int(((both["관측연수"] != both["관측연수_A"])
                 | ((both["개화기간_중앙값"] - both["개화기간_중앙값_A"]).abs() > 1e-9)).sum())
    msgs.append(f"개화시작과 낙화가 함께 {C.MIN_YEARS}년 이상인 조합: 이 표 {len(mine)}개, A의 표 {len(Hh)}개, "
                f"한쪽에만 있음 {only}, 해 수나 기간 중앙값 다름 {bad_k}")
    ok = bad_n == bad_m == bad_s == only == bad_k == 0
    return ok, msgs

def main():
    if C.mode() != "정제본":
        print("정제본(data/interim)이 없어 서비스 표를 만들지 않는다. 함수는 집계표 근사로 돈다.")
        return
    P = pd.read_csv(C.PHENO_SITE)
    H = history(P)
    S = combos(H)
    B = pairs(H)
    G = gaps(H, B)
    fpath = C.folder("features") / C.FEATURES.name
    Z = M.dataset(fpath)
    SH = shift(Z)
    D = decision()
    M1T, M1X = lines(Z, fpath)
    out = C.SERVICE
    out.mkdir(parents=True, exist_ok=True)
    tables = {"combo_stats.csv": S, "history.csv": H, "bloom_pairs.csv": B, "gaps.csv": G, "shift.csv": SH,
              "decision.csv": D, "m1.csv": M1T, "m1_inputs.csv": M1X}
    for name, T in tables.items():
        T.to_csv(out / name, index=False, encoding="utf-8-sig")
        print(f"저장: {(out / name).relative_to(C.ROOT)} {T.shape}")
    print(SH.to_string(index=False))
    print(D[["항목", "현상", "판정", "값", "입력"]].to_string(index=False))
    ok, msgs = parity(S, B)
    for m in msgs:
        print(("" if ok else "!! ") + m)
    A = pd.read_csv(C.TABLES / "eda_year_anomaly.csv").sort_values("연도")
    for phen, col in (("개화", C.STAGE_SPRING), ("단풍", C.STAGE_AUTUMN)):
        mine = SH.loc[SH["현상"] == phen, "최근5년_편차"]
        if len(mine):
            print(f"{phen} 최근 5년 편차: 이 표 {float(mine.iloc[0]):+.2f}일, A의 eda_year_anomaly {A[col].tail(C.RECENT_N).mean():+.2f}일"
                  + (" (A는 16종, 이 표는 단풍 5종)" if phen == "단풍" else ""))
    if not ok:
        print("!! A의 집계표와 다른 곳이 있다. 정제본 판을 A와 맞춰 볼 것")

if __name__ == "__main__":
    main()
