# 7단계 시계 모델(M2) 검증 후 비교표와 판정표 저장

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import clock as K
import features as F
import mconfig as C
import models as M

REFS = list(C.REF_DATES)
NEAR = "꽃 피기 1~3주 전" # 기준일 넷에서 평년 개화일이 1~3주 뒤인 줄만 모은 채점
LOYO, FORWARD = M.SCHEMES[:2]
M0 = M.spec("M0")
M0C = M.spec("M0", label="기준일에 자름")
MRC = M.spec("M0r", label="기준일에 자름")
M2C = M.spec("M2", label="시계만")
M2 = M.spec("M2", label="시계와 관측")
PRED = {"M0": M0C, "M0r": MRC, "시계만": M2C, "시계와 관측": M2} # 시계 모듈의 예측 이름별 모델 정의

# 비교 목록 생성
# name은 표에 적는 기준일 이름
# ref를 주면 그날까지 이미 핀 줄 제외
# 한 해씩 빼기는 자른 M0와 비교하고 나머지 검증은 자른 M0r과도 비교
def items(scheme, name, ref=None):
    m0, m0c, mrc, m2c, m2 = (dict(s, 기준일=name) for s in (M0, M0C, MRC, M2C, M2))
    it = [M.item(m0, m0, ref), M.item(m0c, m0, ref), M.item(m2c, m0c, ref), M.item(m2, m0c, ref)]
    if scheme != LOYO:
        it += [M.item(mrc, m0c, ref), M.item(m2c, mrc, ref), M.item(m2, mrc, ref)]
    return it

# 검증 한 번의 학습 해와 검증 해와 학습 줄과 검증 줄 표시 생성
def marks(B, scheme, tr, te):
    years = np.array(C.YEARS)
    tey = np.isin(years, te["연도"].unique())
    oky = ~tey if scheme == LOYO else years < te["연도"].min()
    trb, teb = np.zeros(len(B), bool), np.zeros(len(B), bool)
    trb[tr.index], teb[te.index] = True, True
    return oky, tey, trb, teb

# 검증 한 번의 한 기준일 예측을 줄 단위 긴 표로 변환
# 이미핌은 기준일까지 핀 줄 표시
# 가까운줄은 평년 개화일이 기준일 1~3주 뒤인 줄 표시
def lines(B, te, out, ref, name):
    i = te.index.to_numpy()
    base = B.loc[i, M.ROW].copy()
    doy = np.array([F.refdoy(ref, y) for y in base["연도"]])
    lead = out["base"][i] - doy
    base["기준일"], base["fold"], base["half"] = ref, name, np.nan
    base["조합평균"] = out["base"][i]
    base["상태"] = np.array(K.STATES)[out["state"][i]]
    base["이미핌"] = base["날짜숫자"].to_numpy() <= doy
    base["가까운줄"] = (lead > C.M2_NEAR[0]) & (lead <= C.M2_NEAR[1])
    parts = [base.assign(spec=M0["id"], pred=out["base"][i], low=np.nan, high=np.nan)]
    for k, (mid, low, high) in out["pred"].items():
        parts.append(base.assign(spec=PRED[k]["id"], pred=mid[i], low=low[i], high=high[i]))
    return pd.concat(parts, ignore_index=True)

# 누출 점검
# 검증 해의 기준일 뒤 관측 날짜와 그 뒤 해의 관측 날짜를 999로 바꿔도 예측이 그대로인지 확인
# 같은 범위의 기온을 999도로 바꿔도 예측이 그대로인지 확인
# 관측 쪽과 기온 쪽에서 어긋난 값 수 반환
def leak(L, H, W, out, oky, tey, trb, teb, ref, lag):
    et, d, rt = L["year"], L["date"], H["ref"][ref]
    late = ~oky & ~tey # 검증 해보다 뒤 해
    obs = K.fold(L, H, np.where(late[et] | (tey[et] & (d > rt[et])), 999.0, d), oky, trb, ref, lag)
    hot = W.copy()
    hot[:, late, :] = 999.0
    for t in np.flatnonzero(tey):
        hot[:, t, rt[t] + 1:] = 999.0
    air = K.fold(L, K.heat(hot, [ref]), d, oky, trb, ref, lag)
    bad = [0, 0]
    for k, v in out["pred"].items():
        for n, other in enumerate((obs, air)):
            bad[n] += sum(int((~np.isclose(a[teb], b[teb])).sum()) for a, b in zip(v, other["pred"][k]))
    return bad

# 범위 점검
# 예측이 기준일 뒤인지와 아래 가운데 위 순서인지 확인
# 이득과 폭 배수가 정해 둔 범위 안인지 확인
# 어긋난 값 수 반환
def bounds(out, teb, doy):
    odd = 0
    for mid, low, high in out["pred"].values():
        odd += int(((mid[teb] <= doy[teb]) | (low[teb] > mid[teb]) | (mid[teb] > high[teb])).sum())
    est = out["est"]
    odd += sum(not C.M2_GAIN[0] - 1e-9 <= est[f"{k} 이득"] <= C.M2_GAIN[1] + 1e-9 for k in K.SIGNALS)
    odd += sum(not C.M2_WIDTH[0] - 1e-9 <= est[f"폭 배수 {k}"] <= C.M2_WIDTH[1] + 1e-9 for k in K.STATES)
    return odd

# 비교표 줄과 해별 줄 생성
# 채점은 모델 모듈의 score 그대로 사용
# 80% 범위는 위아래 길이가 달라 포함률만 여기서 계산
# cut은 묶음 채점에서 이미 피어 뺀 줄
def table(P, its, meta, groups, cut=None):
    rows, yrs = [], []
    for gname, gmask in groups.items():
        for it in its:
            res, yr = M.score(P, it["sid"], it["base"], ref=it["ref"], mask=gmask)
            if res is None:
                continue
            j = P[(P["spec"] == it["sid"]) & ~P["이미핌"]]
            j = j if gmask is None else j[gmask(j)]
            if len(j) != res["줄수"]:
                sys.exit("!! 채점한 줄 수가 맞지 않음")
            if j["low"].notna().all():
                res["구간80_포함률"] = float(((j["low"] <= j["y"]) & (j["y"] <= j["high"])).mean())
            if cut is not None:
                c = cut[cut["spec"] == it["sid"]]
                res["제외줄수"] = int(len(c) if gmask is None else gmask(c).sum())
            info = {**meta, "기준일": it["기준일"], "모델": it["모델"], "입력": it["입력"], "기준모델": it["기준모델"],
                    "종집단": gname}
            rows.append({**info, **res})
            yrs.append(yr.assign(**info))
    return rows, yrs

# 판정표 생성
# 기준일 넷과 묶음 채점마다 두 검증을 모두 통과해야 채택
# 한 해씩 빼기는 자른 M0와 비교하고 앞으로 가는 검증은 자른 M0r과 비교
# 반올림하기 전 값으로 판정
def decision(rows, lab):
    def entry(scheme, name, base):
        want = (scheme, name, M2["모델"], M2["입력"], M.item(M2, base)["기준모델"], "전체")
        hit = [r for r in rows if tuple(r[k] for k in ("검증방식", "기준일", "모델", "입력", "기준모델", "종집단")) == want]
        return hit[0] if hit else None

    plan = ((LOYO, M0C, C.RULE_WINS_LOYO, len(C.LOYO_YEARS)), (FORWARD, MRC, C.RULE_WINS_FORWARD, len(C.FORWARD_YEARS)))
    keep = ("줄수", "해수", "기준대비_MAE차", "MAE차_95하한", "MAE차_95상한", "이긴_해수")
    dec = []
    for name in REFS + [NEAR]:
        got = [entry(scheme, name, base) for scheme, base, _, _ in plan]
        passed = [M.verdict(r, wins) for r, (_, _, wins, _) in zip(got, plan)]
        ok = all(passed)
        if name == NEAR:
            use = "기준일마다 평년 개화일이 1~3주 뒤인 조합의 예측에 M2를 쓴다" if ok else "이 범위만 따로 쓰지 않는다"
        else:
            use = f"{name}에 아직 안 핀 조합의 예측에 M2를 쓴다" if ok else f"{name}에는 기준 모델만 쓴다"
        for r, good, (scheme, base, wins, n) in zip(got, passed, plan):
            dec.append({"현상": "개화", "항목": f"M2({name})", "비교": f"M2({M2['입력']}) 대 {M.item(M2, base)['기준모델']}",
                        "목표단계": C.STAGE_SPRING, "검증방식": scheme, "기준일": name, "자료판": lab,
                        **{k: (r or {}).get(k) for k in keep},
                        "규칙": f"MAE {C.RULE_MAE}일 이상 감소, {n}해 중 {wins}해 이상",
                        "검증_통과": "통과" if good else "미달", "판정": "채택" if ok else "기각", "기능에서_쓰는_값": use})
    D = pd.DataFrame(dec)
    for c in ("기준대비_MAE차", "MAE차_95하한", "MAE차_95상한"):
        D[c] = pd.to_numeric(D[c], errors="coerce").round(2)
    for c in ("줄수", "해수", "이긴_해수"):
        D[c] = pd.to_numeric(D[c], errors="coerce").round().astype("Int64")
    return D

def main():
    mode = C.mode()
    if mode != "정제본":
        print("자료판:", mode, "| 시계 모델(M2)은 정제본에서만 돌린다")
        return
    Z = pd.read_csv(C.folder("work") / C.TRAIN.name)
    B = Z[(Z["현상"] == "개화") & (Z["목표단계"] == C.STAGE_SPRING)].reset_index(drop=True)
    E = K.events()
    L = K.layout(E, B)
    daily, src = F.weather()
    if C.DAEGU_ADJUST:
        daily, _ = F.correction(daily)
    lab = "대구 보정" if C.DAEGU_ADJUST else "대구 원값"
    W = K.temps(daily, L["sites"])
    H = K.heat(W, REFS)
    doys = {ref: H["ref"][ref][L["year"][L["target"]]] for ref in REFS}
    print("자료판:", mode, "|", lab, "| 기상 일자료:", src)
    print(f"목표 줄 {len(B)}줄 조합 {B['조합'].nunique()}개 | 사건 {len(E)}줄 계열 {L['series'].max() + 1}개")

    groups = M.cohorts(B)
    meta = {"현상": "개화", "목표단계": C.STAGE_SPRING}
    rows, yrs, preds, ests = [], [], [], []
    bad, odd = [0, 0], 0
    for scheme in M.SCHEMES:
        parts = []
        for name, tr, te in M.folds(B, scheme):
            if te.empty or tr.empty:
                continue
            assert not set(tr["연도"]) & set(te["연도"]), "학습과 검증에 같은 해가 들어갔다"
            oky, tey, trb, teb = marks(B, scheme, tr, te)
            lag = None if scheme == LOYO else M.drift(tr)
            for ref in REFS:
                out = K.fold(L, H, L["date"], oky, trb, ref, lag)
                parts.append(lines(B, te, out, ref, name))
                ests.append({"검증방식": scheme, "fold": name, "기준일": ref, **out["est"]})
                bad = [a + b for a, b in zip(bad, leak(L, H, W, out, oky, tey, trb, teb, ref, lag))]
                odd += bounds(out, teb, doys[ref])
        P = pd.concat(parts, ignore_index=True)
        P["검증방식"] = scheme
        info = {**meta, "검증방식": scheme, "자료판": lab}
        for ref in REFS:
            r, y = table(P[P["기준일"] == ref], items(scheme, ref, ref), info, groups)
            rows, yrs = rows + r, yrs + y
        near = P[P["가까운줄"]]
        if near.duplicated(["spec", "조합", "연도"]).any():
            sys.exit("!! 묶음 채점에 같은 줄이 두 번 들어감")
        r, y = table(near[~near["이미핌"]], items(scheme, NEAR), info, groups, cut=near[near["이미핌"]])
        rows, yrs, preds = rows + r, yrs + y, preds + [P]

    G = pd.DataFrame(ests)
    G.columns = [c.replace(" ", "_") for c in G.columns]
    loose = int((~G["추정_끝"]).sum())
    print("누출 점검: 검증 해의 기준일 뒤 관측을 999로 바꾼 뒤 달라진 예측", bad[0], "개 | 기온을 999도로 바꾼 뒤", bad[1], "개")
    print("범위 점검: 기준일보다 이르거나 순서가 뒤집힌 예측과 범위를 벗어난 이득", odd, "개 | 추정이 끝나지 않은 검증", loose, "번")
    if bad[0] or bad[1]:
        sys.exit("!! 누출 점검 실패")
    if odd or loose:
        sys.exit("!! 범위 점검 실패")

    T = M.frame(rows, M.COMP_COLS)
    Y = M.frame([q for d in yrs for q in d.to_dict("records")], M.YEAR_COLS)
    Y = Y.drop_duplicates(subset=[c for c in M.YEAR_COLS if c not in ("줄수", "MAE", "편향")])
    D = decision(rows, lab)

    tdir, mdir = C.folder("tables"), M.folder()
    T.to_csv(tdir / "model_challenge.csv", index=False, encoding="utf-8-sig")
    Y.to_csv(tdir / "model_challenge_by_year.csv", index=False, encoding="utf-8-sig")
    D.to_csv(tdir / "model_challenge_decision.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_csv(mdir / "challenge_predictions.csv", index=False, encoding="utf-8-sig")
    G.to_csv(mdir / "challenge_gains.csv", index=False, encoding="utf-8-sig")
    print("저장:", (tdir / "model_challenge.csv").relative_to(C.ROOT), T.shape, "/", "model_challenge_by_year.csv", Y.shape,
          "/", "model_challenge_decision.csv", D.shape)

    show = T[T["종집단"] == "전체"]
    cols = ["검증방식", "기준일", "모델", "입력", "기준모델", "줄수", "제외줄수", "MAE", "기준대비_MAE차", "MAE차_95하한",
            "MAE차_95상한", "이긴_해수", "해수", "구간80_포함률"]
    print(show[cols].to_string(index=False))
    mean = G[G["검증방식"] == LOYO].drop(columns=["검증방식", "fold", "추정_끝"]).groupby("기준일", sort=False).mean()
    print("한 해씩 빼기 검증의 추정값 평균")
    print(mean.round(2).to_string())
    print(D[["항목", "검증방식", "줄수", "기준대비_MAE차", "이긴_해수", "해수", "검증_통과", "판정"]].to_string(index=False))

if __name__ == "__main__":
    main()
