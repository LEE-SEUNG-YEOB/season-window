# 4단계 M1과 랜덤포레스트 검증 후 비교표와 채택표 저장

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import mconfig as C
import models as M

M0 = M.spec("M0")
M1 = M.spec("M1", C.M1_MAIN, ref="2월 말")
M1S = [M.spec("M1", x, ref="2월 말") for x in C.M1_SENS]
M1U = M.spec("M1", C.M1_UPPER, ref="상한(3월 말까지 다 앎)")
M0F = dict(M0, 기준일="2월 말") # 2월 말 줄과 같은 줄에서 잰 M0

def spring(Z):
    st = C.STAGE_SPRING if C.mode() == "정제본" else "수목원 연도 효과(집계표)"
    return Z[(Z["현상"] == "개화") & (Z["목표단계"] == st)].copy(), st

def forest(Z):
    wide = [c for c in C.RF_WIDE if Z[c].notna().any()]
    lab = "넓은 입력: " + ", ".join(wide) + " 편차 + 조합 평균 + 종"
    return (M.spec("RF", wide, ref="2월 말", label=lab),
            M.spec("RF", [C.M1_MAIN], ref="2월 말", label=f"{C.M1_MAIN} 편차 + 조합 평균 + 종"))

def loyo(with_rf, rf=()):
    it = [M.item(M0, M0), M.item(M0F, M0, ref="2월 말"), M.item(M1, M0, ref="2월 말")]
    it += [M.item(s, M0, ref="2월 말") for s in M1S] + [M.item(M1U, M0)]
    if with_rf:
        it += [M.item(s, M0, ref="2월 말") for s in rf] + [M.item(rf[0], M1, ref="2월 말")]
    return it

def main():
    mode = C.mode()
    fdir = C.folder("features")
    main_lab, other_lab = ("대구 보정", "대구 원값") if C.DAEGU_ADJUST else ("대구 원값", "대구 보정")
    other_file = C.FEATURES_RAW.name if C.DAEGU_ADJUST else "weather_features_daegu_adj.csv"
    Z, st = spring(M.dataset(fdir / C.FEATURES.name))
    Zo, _ = spring(M.dataset(fdir / other_file))
    species = sorted(Z["종"].unique())
    with_rf = mode == "정제본"
    rf = forest(Z) if with_rf else ()
    groups = M.cohorts(Z)
    meta = {"현상": "개화", "목표단계": st}
    rows, yrs, preds, slopes = [], [], [], []

    # 주 자료판 검증
    hitems = [M.item(M0F, M0, ref="2월 말"), M.item(M1, M0, ref="2월 말"), M.item(M1U, M0)]
    hitems += [M.item(rf[0], M0, ref="2월 말")] if with_rf else []
    plan = (("한 해씩 빼기", [M0, M1, *M1S, M1U, *rf], loyo(with_rf, rf)),
            ("앞으로 가는 검증", [M0, M1], [M.item(M1, M0, ref="2월 말")]),
            ("최근 4년 떼어 두기", [M0, M1, M1U, *rf[:1]], hitems))
    for scheme, specs, items in plan:
        P, S = M.cv(Z, scheme, specs, species)
        r, y = M.comparison(P, items, {**meta, "검증방식": scheme, "자료판": main_lab}, groups)
        rows, yrs, preds, slopes = rows + r, yrs + y, preds + [P], slopes + [S.assign(검증방식=scheme)]

    # 자료판 민감도(한 해씩 빼기에서 M0와 M1만)
    daegu, caution = C.DAEGU["site"], set(C.WEATHER_CAUTION)
    variants = {other_lab: Zo, f"{daegu} 제외": Z[Z["수목원"] != daegu],
                "기상주의 줄 제외": Z[[(s, yv) not in caution for s, yv in zip(Z["수목원"], Z["연도"])]]}
    for lab, Zv in variants.items():
        P, _ = M.cv(Zv, "한 해씩 빼기", [M0, M1, *M1S, M1U], species)
        r, y = M.comparison(P, loyo(False), {**meta, "검증방식": "한 해씩 빼기", "자료판": lab})
        rows, yrs = rows + r, yrs + y

    # 누출 점검
    bad = M.leak(Z, "한 해씩 빼기", [M0, M1, *M1S, M1U])
    bad += M.leak(Z, "앞으로 가는 검증", [M0, M.spec("M0r"), M1])
    if with_rf:
        one = Z[Z["연도"].isin(sorted(Z["연도"].unique())[-2:])] # RF는 시간이 들어 두 해만 확인
        bad += M.leak(pd.concat([Z[~Z.index.isin(one.index)], one]), "최근 4년 떼어 두기", rf[:1], species)
    inputs = [C.M1_MAIN, *C.M1_SENS, C.M1_UPPER, *C.RF_WIDE]
    own = [c for c in inputs if any(k in c for k in ("날짜", "개화", "낙화", "단풍"))]
    print("누출 점검: 검증 해 날짜를 999로 바꾼 뒤 달라진 예측", bad, "줄 | 그해 관측이 든 입력", own or "없음")
    if bad or own:
        sys.exit("!! 누출 점검 실패")

    # 3단계 결과와 합쳐 저장
    mdir = M.folder()
    base_rows = pd.read_csv(mdir / "comparison_baseline.csv")
    base_years = pd.read_csv(mdir / "cv_by_year_baseline.csv")
    T = M.frame(pd.concat([M.frame(base_rows, M.COMP_COLS), M.frame(rows, M.COMP_COLS)], ignore_index=True), M.COMP_COLS)
    Y = pd.concat([M.frame(base_years, M.YEAR_COLS),
                   M.frame([q for d in yrs for q in d.to_dict("records")], M.YEAR_COLS)], ignore_index=True)
    key = ["현상", "목표단계", "검증방식", "기준일", "모델", "입력", "기준모델", "종집단", "자료판"]
    T = T.drop_duplicates(subset=key) # 3단계와 겹치는 M0 줄은 3단계 것 유지
    Y = M.frame(Y, M.YEAR_COLS).drop_duplicates(subset=[k for k in key if k != "기준모델"] + ["연도"])

    def entry(scheme, model, inp, base="M0", ref="2월 말"):
        q = T[(T["목표단계"] == st) & (T["검증방식"] == scheme) & (T["모델"] == model) & (T["입력"] == inp)
              & (T["기준모델"] == base) & (T["종집단"] == "전체") & (T["자료판"] == main_lab) & (T["기준일"] == ref)]
        return None if q.empty else q.iloc[0].to_dict()

    dec = pd.read_csv(mdir / "decision_baseline.csv").to_dict("records")
    m1 = entry("한 해씩 빼기", "M1", C.M1_MAIN)
    ok1 = M.verdict(m1, C.RULE_WINS_LOYO)
    rule = f"MAE {C.RULE_MAE}일 이상 감소, {len(C.LOYO_YEARS)}해 중 {C.RULE_WINS_LOYO}해 이상"
    keep = ("줄수", "해수", "기준대비_MAE차", "MAE차_95하한", "MAE차_95상한", "이긴_해수")
    dec.append({"현상": "개화", "항목": "M1(2월 말 예측)", "비교": f"M1({C.M1_MAIN}) 대 M0", "목표단계": st,
                "검증방식": "한 해씩 빼기", "기준일": "2월 말", "자료판": main_lab, **{k: m1[k] for k in keep},
                "규칙": rule, "판정": "채택" if ok1 else "기각",
                "기능에서_쓰는_값": "2월 말 예측(M1)" if ok1 else "2월 기온으로는 뚜렷이 줄지 않았다. 기준 모델만 쓴다"})
    if with_rf:
        rfr = entry("한 해씩 빼기", "RF", rf[0]["입력"], base=f"M1({C.M1_MAIN})")
        okr = M.verdict(rfr, C.RULE_WINS_LOYO)
        dec.append({"현상": "개화", "항목": "RF(넓은 입력)", "비교": f"RF 대 M1({C.M1_MAIN})", "목표단계": st,
                    "검증방식": "한 해씩 빼기", "기준일": "2월 말", "자료판": main_lab, **{k: rfr[k] for k in keep},
                    "규칙": rule, "판정": "후보" if okr else "기각",
                    "기능에서_쓰는_값": "후보로 적고 팀이 정한다" if okr else "단순한 쪽(M0 또는 M1)을 쓴다"})
    else:
        dec.append({"현상": "개화", "항목": "RF(넓은 입력)", "판정": "보류", "기능에서_쓰는_값": "정제본(강수, 일조 포함)에서 돌린다"})
    D = pd.DataFrame(dec)
    for c in ("기준대비_MAE차", "MAE차_95하한", "MAE차_95상한"):
        D[c] = pd.to_numeric(D[c], errors="coerce").round(2)
    for c in ("줄수", "해수", "이긴_해수"):
        D[c] = pd.to_numeric(D[c], errors="coerce").round().astype("Int64")

    tdir = C.folder("tables")
    T.to_csv(tdir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    Y.to_csv(tdir / "model_cv_by_year.csv", index=False, encoding="utf-8-sig")
    D.to_csv(tdir / "model_decision.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_csv(mdir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(slopes, ignore_index=True).to_csv(mdir / "fold_slopes.csv", index=False, encoding="utf-8-sig")
    print("자료판:", mode, "| 저장:", (tdir / "model_comparison.csv").relative_to(C.ROOT), T.shape, "/",
          (tdir / "model_cv_by_year.csv").name, Y.shape, "/", (tdir / "model_decision.csv").name)

    show = T[(T["목표단계"] == st) & (T["종집단"] == "전체")]
    cols = ["검증방식", "기준일", "모델", "입력", "기준모델", "자료판", "줄수", "제외줄수", "MAE", "기준대비_MAE차",
            "MAE차_95하한", "MAE차_95상한", "이긴_해수", "해수", "구간80_포함률"]
    print(show[cols].to_string(index=False))
    print(D[["현상", "항목", "기준대비_MAE차", "이긴_해수", "해수", "판정", "기능에서_쓰는_값"]].to_string(index=False))

if __name__ == "__main__":
    main()
