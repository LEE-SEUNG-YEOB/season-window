# 2단계는 학습 표를 만든다
# 해별 날짜가 든 줄 단위 표라 data/processed 아래에만 둔다(올리지 않는다)
#
# 정제본 판은 봄꽃 11종의 개화50%이상과 단풍 5종의 단풍50%이상을 쓴다
# 어느 쪽이든 그 단계가 8년 이상 있는 조합만 쓴다
# 집계표 판은 A의 수목원 연도 효과 표와 전국 연도 편차 표를 쓴다
# 이상치의심은 표시로만 옮기고 빼지 않는다(A와 같은 처리)

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import features as F
import mconfig as C

KEY = ["수목원", "종", "연도"]
COLS = ["현상", "목표단계", "조합", "수목원", "종", "종집단", "연도", "y", "날짜숫자", "개화시작", "낙화",
        "나무수", "이상치의심", "기상주의"]


def group_of(sp):
    if sp in C.EARLY_SPECIES:
        return "이른 종"
    if sp in C.LATE_SPECIES:
        return "늦은 종"
    return "단풍 5종" if sp in C.AUTUMN_SPECIES else "기타"


def from_clean():
    P = pd.read_csv(C.PHENO_SITE)
    dup = int(P.duplicated(KEY + ["현상"]).sum())
    if dup:
        sys.exit(f"!! 같은 수목원 종 연도 현상 줄이 {dup}개 겹친다. 정제본을 확인할 것")
    P = P.dropna(subset=["날짜숫자"])

    def stage(name):
        return P[P["현상"] == name].set_index(KEY)["날짜숫자"]

    out = []
    for phen, st, species in (("개화", C.STAGE_SPRING, C.SPRING_SPECIES), ("단풍", C.STAGE_AUTUMN, C.AUTUMN_SPECIES)):
        x = P[(P["현상"] == st) & P["종"].isin(species)].copy()
        x = x[x.groupby(["수목원", "종"])["연도"].transform("nunique") >= C.MIN_YEARS].set_index(KEY)
        x["개화시작"] = stage(C.STAGE_START).reindex(x.index) if phen == "개화" else float("nan")
        x["낙화"] = stage(C.STAGE_END).reindex(x.index) if phen == "개화" else float("nan")
        x = x.reset_index()
        x["현상"], x["목표단계"] = phen, st
        x["조합"] = x["수목원"] + "|" + x["종"]
        x["종집단"] = x["종"].map(group_of)
        x["y"] = x["날짜숫자"]
        if "나무수" not in x:
            x["나무수"] = float("nan")
        if "이상치의심" not in x:
            x["이상치의심"] = False
        out.append(x)
    return pd.concat(out, ignore_index=True)


def from_aggregates():
    ye = pd.read_csv(C.TABLES / "review_year_effect.csv").set_index("수목원").drop(index="전체")
    ye = ye.stack().rename("y").reset_index().rename(columns={"level_1": "연도"})
    ye["연도"] = ye["연도"].astype(int)
    ye = ye[~ye.set_index(["수목원", "연도"]).index.isin(C.AGG_DROP)]
    ye = ye.assign(현상="개화", 목표단계="수목원 연도 효과(집계표)", 종="전체", 종집단="전체")
    ye["조합"] = ye["수목원"]
    an = pd.read_csv(C.TABLES / "eda_year_anomaly.csv")
    nat = []
    for phen, st in (("개화", C.STAGE_SPRING), ("단풍", C.STAGE_AUTUMN)):
        nat.append(pd.DataFrame({"현상": phen, "목표단계": "전국 연도 편차(집계표)", "조합": "전국", "수목원": "전국",
                                 "종": "전체", "종집단": "전체", "연도": an["연도"], "y": an[st]}))
    Z = pd.concat([ye] + nat, ignore_index=True)
    for c in ("날짜숫자", "개화시작", "낙화", "나무수"):
        Z[c] = float("nan")
    Z["이상치의심"] = False
    return Z


def main():
    mode = C.mode()
    Z = from_clean() if mode == "정제본" else from_aggregates()
    Z["기상주의"] = [(s, y) in set(C.WEATHER_CAUTION) for s, y in zip(Z["수목원"], Z["연도"])]
    Z = Z.dropna(subset=["y"])[COLS].sort_values(["현상", "목표단계", "조합", "연도"]).reset_index(drop=True)
    path = C.out_dir("work") / C.TRAIN.name
    Z.to_csv(path, index=False, encoding="utf-8-sig")
    print("자료판:", mode, "| 저장:", path.relative_to(C.ROOT), Z.shape)

    for (phen, st), g in Z.groupby(["현상", "목표단계"]):
        print(f"- {phen} {st}: 조합 {g['조합'].nunique()}개 {len(g)}줄, 해 {g['연도'].nunique()}개,"
              f" 이상치의심 {int(g['이상치의심'].sum())}줄(빼지 않음), 기상주의 {int(g['기상주의'].sum())}줄")
        if mode != "정제본":
            continue
        for grp, h in g.groupby("종집단"):
            print(f"    {grp}: 조합 {h['조합'].nunique()}개 {len(h)}줄")
        if phen == "개화":
            for name in C.REF_DATES:
                doy = g["연도"].map(lambda y: F.ref_doy(name, y))
                early = g["y"] <= doy
                print(f"    기준일 {name}: 이미 핀 줄 {int(early.sum())}, 그런 해가 있는 조합 {g.loc[early, '조합'].nunique()}개")
            paired = g.dropna(subset=["개화시작", "낙화"]).groupby("조합").size()
            few = sorted(set(g["조합"]) - set(paired[paired >= C.MIN_YEARS].index))
            print(f"    학습표 안(개화50%이상이 있는 해)에서 개화시작과 낙화가 함께 {C.MIN_YEARS}년 이상인 조합"
                  f" {int((paired >= C.MIN_YEARS).sum())}개, 모자란 조합 {few}"
                  " (A의 87개는 개화50%이상과 상관없이 개화기간이 0보다 큰 해로 센 값이라 정의가 다르다)")
    if mode == "정제본":
        sp = Z[Z["현상"] == "개화"]
        want = {"조합": 91, "줄": 1311}
        got = {"조합": sp["조합"].nunique(), "줄": len(sp)}
        print("설계서 숫자와 비교(개화):", got, "설계서", want, "같음" if got == want else "다름. 설계서 숫자를 고칠 것")


if __name__ == "__main__":
    main()
