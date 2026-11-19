# 6단계는 함수 파일이 읽는 서비스 표를 data/processed/service에 만든다
#
# 정제본이 있을 때만 만든다
# 해별 날짜가 든 표라 저장소에 올리지 않는다(data/processed는 무시 목록)
#
# 정의는 A의 집계와 같다(analysis 폴더의 5번 탐색 분석 파일)
# 개화기간 = 낙화90~100% - 개화시작(0보다 큰 해만)
# 꽃이 핀 날 = 개화시작 <= 날 <= 낙화90~100%
#
# 만드는 표는 아래 여덟 개다
# 파일 이름은 main 함수의 tables에 같은 순서로 적혀 있다
# 조합 통계 표는 수목원 x 종 x 단계의 관측연수와 평균과 표준편차다(범위와 분포 근사에 쓴다)
# 해별 날짜 표는 기능 1의 history와 그해 실제 날짜숫자에 쓴다
# 개화 짝 표는 개화시작과 낙화가 함께 있는 해다(기능 2와 3의 지난 해 세기에 쓴다)
# 간격 표는 개화시작~개화50%이상 간격과 꽃 시기 길이의 중앙값이다(조합별과 종별)
# 편차 표는 전국 최근 5년 편차다(M0r에 쓰는 값)
# 판정 표는 고르는 규칙의 판정과 가정 입력 기울기다
# M1 표 두 개는 M1 계수와 수목원 해별 입력이다(M1이 채택됐을 때만 함수가 쓴다)
#
# 끝에 A의 집계표와 견준다(설계서 10절 2번과 3번)

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


def combo_stats(H):
    S = H.groupby(["수목원", "종", "현상"])["날짜숫자"].agg(
        관측연수="count", 평균="mean", 표준편차="std", 가장이른="min", 가장늦은="max").reset_index()
    return S.round({"평균": 3, "표준편차": 3})


def _wide(H, stages):
    W = H[H["현상"].isin(stages)].pivot(index=KEY, columns="현상", values="날짜숫자")
    return W.reindex(columns=stages).dropna().reset_index()


def bloom_pairs(H):
    W = _wide(H, [C.STAGE_START, C.STAGE_END]).rename(columns={C.STAGE_START: "개화시작", C.STAGE_END: "낙화"})
    W["개화기간"] = W["낙화"] - W["개화시작"]
    return W[W["개화기간"] > 0][KEY + ["개화시작", "낙화", "개화기간"]]


def gaps(H, B):
    W = _wide(H, [C.STAGE_START, C.STAGE_SPRING])
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
        an = (g["y"] - g.groupby("조합")["y"].transform("mean")).groupby(g["연도"]).mean().sort_index()
        last = an.tail(C.RECENT_N)
        rows.append({"현상": phen, "목표단계": st, "시작연도": int(last.index.min()), "끝연도": int(last.index.max()),
                     "최근5년_편차": round(M.recent_shift(g), 2), "조합수": g["조합"].nunique(), "줄수": len(g)})
    return pd.DataFrame(rows)


def decision():
    tdir = C.out_dir("tables")
    D = pd.read_csv(tdir / "model_decision.csv")
    S = pd.read_csv(tdir / "model_species_slope.csv")
    rows = [{"항목": str(r["항목"]).split("(")[0], "현상": r["현상"], "판정": r["판정"], "값": None, "하한95": None,
             "상한95": None, "입력": None, "비고": r.get("기능에서_쓰는_값")} for _, r in D.iterrows()]
    s = S[S["쓰임"] == "가정 입력 기울기"].iloc[0]
    rows.append({"항목": "가정 입력 기울기", "현상": "개화", "판정": "", "값": s["기울기_일per도"], "하한95": s["하한95"],
                 "상한95": s["상한95"], "입력": s["입력"], "비고": f"{s['자료판']}, 조합 {s['조합수']}개 {s['줄수']}줄"})
    return pd.DataFrame(rows)


def m1_tables(Z, features_path):
    t = Z[(Z["현상"] == "개화") & (Z["목표단계"] == C.STAGE_SPRING)]
    b, mx, sd = M.m1_fit(t, C.M1_MAIN)
    g = t.groupby("조합")
    T = pd.DataFrame({"평균": g["y"].mean(), "해수": g["y"].count(), "평년_기온": mx, "잔차_표준편차": sd}).reset_index()
    T[["수목원", "종"]] = T["조합"].str.split("|", expand=True)
    T["입력"], T["기울기"] = C.M1_MAIN, round(b, 3)
    T = T[["수목원", "종", "입력", "기울기", "평균", "평년_기온", "잔차_표준편차", "해수"]].round(3)
    X = pd.read_csv(features_path)[["수목원", "연도", C.M1_MAIN]]
    return T, X


# A의 집계표와 견준다
# 다르면 정제본 판이 다르거나 정의가 어긋난 것이다
def compare_with_a(S, B):
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
    msgs.append(f"개화시작과 낙화가 함께 8년 이상인 조합: 이 표 {len(mine)}개, A의 표 {len(Hh)}개, "
                f"한쪽에만 있음 {only}, 해 수나 기간 중앙값 다름 {bad_k}")
    ok = bad_n == bad_m == bad_s == only == bad_k == 0
    return ok, msgs


def main():
    if C.mode() != "정제본":
        print("정제본(data/interim)이 없어 서비스 표를 만들지 않는다. 함수는 집계표 근사로 돈다.")
        return
    P = pd.read_csv(C.PHENO_SITE)
    H = history(P)
    S = combo_stats(H)
    B = bloom_pairs(H)
    G = gaps(H, B)
    fpath = C.out_dir("features") / C.FEATURES.name
    Z = M.load(fpath)
    SH = shift(Z)
    D = decision()
    M1T, M1X = m1_tables(Z, fpath)
    out = C.SERVICE
    out.mkdir(parents=True, exist_ok=True)
    tables = {"combo_stats.csv": S, "history.csv": H, "bloom_pairs.csv": B, "gaps.csv": G, "shift.csv": SH,
              "decision.csv": D, "m1.csv": M1T, "m1_inputs.csv": M1X}
    for name, T in tables.items():
        T.to_csv(out / name, index=False, encoding="utf-8-sig")
        print(f"저장: {(out / name).relative_to(C.ROOT)} {T.shape}")
    print(SH.to_string(index=False))
    print(D[["항목", "현상", "판정", "값", "입력"]].to_string(index=False))
    ok, msgs = compare_with_a(S, B)
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
