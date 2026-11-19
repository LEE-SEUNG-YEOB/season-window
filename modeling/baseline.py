# 3단계는 기준 모델을 검증하고 M0r을 쓸지 정한다
# M0는 조합 평균이다
# M0r은 조합 평균에 전국 최근 5년 편차를 더한 것이다
#
# 앞으로 가는 검증(2015~2025년 11번)에서 M0와 M0r을 견준다(개화와 단풍)
# 한 해씩 빼기 검증(17번)의 M0도 함께 적는다
# 최근 4년 떼어 두기(2022~2025년)의 M0와 M0r도 함께 적는다
# 결과는 data/processed/models 아래 중간 파일로 두고 4단계가 비교표에 합친다

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import mconfig as C
import models as M

M0, M0R = M.spec("M0"), M.spec("M0r")


def main_label():
    return "대구 보정" if C.DAEGU_ADJUST else "대구 원값"


# M0r을 판정할 때 쓰는 목표단계를 돌려준다
# 정제본 판은 조합 단위 목표단계를 쓴다
# 집계표 판은 개화에 수목원 연도 효과를 쓰고 단풍에 전국 편차를 쓴다
def decision_group(Z, phen):
    sts = Z.loc[Z["현상"] == phen, "목표단계"].unique().tolist()
    for want in (C.STAGE_SPRING, C.STAGE_AUTUMN, "수목원 연도 효과(집계표)", "전국 연도 편차(집계표)"):
        if want in sts:
            return want
    return None


def main():
    mode = C.mode()
    Z = M.load(C.out_dir("features") / C.FEATURES.name)
    lab = main_label()
    rows, yrs, dec = [], [], []
    for (phen, st), g in Z.groupby(["현상", "목표단계"]):
        meta = {"현상": phen, "목표단계": st, "자료판": lab}
        groups = M.species_groups(g)
        for scheme in M.SCHEMES:
            if scheme == "한 해씩 빼기" and (phen == "단풍" or st.startswith("전국")):
                continue  # 단풍과 전국 편차는 앞으로 가는 검증으로만 견준다(5절 2항)
            specs = [M0] if scheme == "한 해씩 빼기" else [M0, M0R]
            P, _ = M.run(g, scheme, specs)
            items = [M.item(M0, M0)] + ([M.item(M0R, M0)] if scheme != "한 해씩 빼기" else [])
            r, y = M.compare(P, items, {**meta, "검증방식": scheme}, groups)
            rows += r
            yrs += y
        if decision_group(Z, phen) == st:
            res = next((x for x in rows if x["현상"] == phen and x["목표단계"] == st and x["검증방식"] == "앞으로 가는 검증"
                        and x["모델"] == "M0r" and x["종집단"] == "전체"), None)
            ok = M.adopt(res, C.RULE_WINS_FORWARD)
            dec.append({"현상": phen, "항목": "M0r(계획 시점 기본값)", "비교": "M0r 대 M0", "목표단계": st,
                        "검증방식": "앞으로 가는 검증", "기준일": "기온 모름", "자료판": lab,
                        **{k: res[k] for k in ("줄수", "해수", "기준대비_MAE차", "MAE차_95하한", "MAE차_95상한", "이긴_해수")},
                        "규칙": f"MAE {C.RULE_MAE}일 이상 감소, {len(C.FORWARD_YEARS)}해 중 {C.RULE_WINS_FORWARD}해 이상",
                        "판정": "채택" if ok else "기각",
                        "기능에서_쓰는_값": "조합 평균 + 전국 최근 5년 편차" if ok else "조합 평균"})
    T = M.tidy(rows, M.COMP_COLS)
    Y = M.tidy([r for d in yrs for r in d.to_dict("records")], M.YEAR_COLS)
    D = pd.DataFrame(dec)
    for c in ("기준대비_MAE차", "MAE차_95하한", "MAE차_95상한"):
        D[c] = D[c].round(2)
    out = M.models_dir()
    T.to_csv(out / "comparison_baseline.csv", index=False, encoding="utf-8-sig")
    Y.to_csv(out / "cv_by_year_baseline.csv", index=False, encoding="utf-8-sig")
    D.to_csv(out / "decision_baseline.csv", index=False, encoding="utf-8-sig")
    print("자료판:", mode, "| 저장:", (out / "comparison_baseline.csv").relative_to(C.ROOT))
    show = T[T["종집단"] == "전체"][["현상", "목표단계", "검증방식", "모델", "줄수", "해수", "MAE", "RMSE", "편향",
                                    "기준대비_MAE차", "이긴_해수", "구간80_포함률"]]
    print(show.to_string(index=False))
    print(D[["현상", "목표단계", "기준대비_MAE차", "이긴_해수", "해수", "판정"]].to_string(index=False))


if __name__ == "__main__":
    main()
