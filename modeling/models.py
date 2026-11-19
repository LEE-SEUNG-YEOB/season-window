# 모델 적합과 검증 틀과 성능 지표를 둔다
#
# 조합 평균과 평년(기온의 조합 평균)은 언제나 검증 한 번마다 그때의 학습 해로만 계산한다(4절 2항 2번)
# 입력은 기상 변수와 조합 평균과 종뿐이다(4절 2항 3번)
# 그해의 다른 관측은 넣지 않는다
# 줄을 무작위로 나누지 않고 해를 통째로 나눈다(4절 2항 4번)

import numpy as np
import pandas as pd
from scipy import stats

import features as F
import mconfig as C

ROW = ["연도", "조합", "수목원", "종", "종집단", "y", "날짜숫자", "기상주의"]
SCHEMES = ("한 해씩 빼기", "앞으로 가는 검증", "최근 4년 떼어 두기")


# 학습 표에 기상 변수를 붙인다(수목원 x 연도)
def load(features_path):
    Z = pd.read_csv(C.out_dir("work") / C.TRAIN.name)
    W = pd.read_csv(features_path).drop(columns=["기상지점", "기상지점명", "대구보정", "기상주의", "결측비율_최대"],
                                        errors="ignore")
    return Z.merge(W, on=["수목원", "연도"], how="left")


# 모델 한 줄의 정의를 만든다
# 모델 종류는 M0 M0r M1 RF 가운데 하나다
def spec(model, x=None, ref="기온 모름", label=None):
    inp = label if label is not None else (x if isinstance(x, str) else ", ".join(x) if x else "")
    return {"id": f"{model}|{inp}", "모델": model, "입력": inp, "x": x, "기준일": ref}


# 모델 적합

# 전국 최근 n년 편차를 낸다
# 해마다 날짜에서 조합 평균을 뺀 값의 평균을 낸다
# 학습 해 가운데 마지막 n해의 그 값을 평균한다
def recent_shift(tr, n=C.RECENT_N):
    mu = tr.groupby("조합")["y"].transform("mean")
    an = (tr["y"] - mu).groupby(tr["연도"]).mean().sort_index()
    return float(an.tail(n).mean())


# 공통 기울기를 구한다
# 조합마다 평균을 뺀 날짜를 조합마다 평균을 뺀 기온에 맞춘다
# 종과 수목원을 범주로 넣은 회귀와 같다
def m1_fit(tr, x):
    t = tr.dropna(subset=[x])
    xx = t[x] - t.groupby("조합")[x].transform("mean")
    yy = t["y"] - t.groupby("조합")["y"].transform("mean")
    b = float((xx * yy).sum() / (xx ** 2).sum())
    mx = t.groupby("조합")[x].mean()
    mu = tr.groupby("조합")["y"].mean()
    r = tr["y"] - tr["조합"].map(mu) - b * (tr[x] - tr["조합"].map(mx))
    sd = r.groupby(tr["조합"]).agg(lambda v: np.sqrt((v.dropna() ** 2).sum() / max(v.notna().sum() - 1, 1)))
    return b, mx, sd


def _rf_matrix(tr, df, inputs, species):
    X = pd.DataFrame(index=df.index)
    for c in inputs:
        X[c + "_편차"] = df[c] - df["조합"].map(tr.groupby("조합")[c].mean())
    X["조합평균"] = df["조합"].map(tr.groupby("조합")["y"].mean())
    for s in species:
        X["종_" + s] = (df["종"] == s).astype(float)
    return X


def rf_predict(tr, te, inputs, species):
    from sklearn.ensemble import RandomForestRegressor
    Xtr = _rf_matrix(tr, tr, inputs, species)
    ytr = tr["y"] - tr["조합"].map(tr.groupby("조합")["y"].mean())
    ok = Xtr.notna().all(axis=1)
    m = RandomForestRegressor(**C.RF_PARAMS).fit(Xtr[ok], ytr[ok])
    Xte = _rf_matrix(tr, te, inputs, species)
    out = pd.Series(np.nan, index=te.index)
    okt = Xte.notna().all(axis=1)
    if okt.any():
        out[okt] = m.predict(Xte[okt]) + Xte.loc[okt, "조합평균"].to_numpy()
    return out


# 검증 줄(te)마다 모델별 예측과 80% 범위의 반폭을 낸다
def predict_fold(tr, te, specs, species):
    g = tr.groupby("조합")["y"]
    mu, sd, n = g.mean(), g.std(ddof=1), g.count()
    base = te["조합"].map(mu)
    n_te = te["조합"].map(n).astype(float)
    tq = pd.Series(stats.t.ppf(0.5 + C.INTERVAL_LEVEL / 2, np.maximum(n_te - 1, 1)), index=te.index)
    infl = np.sqrt(1 + 1 / n_te)
    half0 = tq * te["조합"].map(sd) * infl
    out, slopes = {}, {}
    for s in specs:
        kind = s["모델"]
        if kind == "M0":
            p, h = base, half0
        elif kind == "M0r":
            p, h = base + recent_shift(tr), half0
        elif kind == "M1":
            b, mx, rsd = m1_fit(tr, s["x"])
            p = base + b * (te[s["x"]] - te["조합"].map(mx))
            h = tq * te["조합"].map(rsd) * infl
            slopes[s["id"]] = b
        elif kind == "RF":
            p, h = rf_predict(tr, te, s["x"], species), pd.Series(np.nan, index=te.index)
        else:
            raise ValueError(kind)
        out[s["id"]] = (p, h)
    return out, slopes


# 검증 틀

# 검증 한 번마다 이름과 학습 줄과 검증 줄을 차례로 내놓는다
# 학습에는 검증 해가 절대 들어가지 않는다
def folds(Z, scheme):
    if scheme == "한 해씩 빼기":
        for y in sorted(Z["연도"].unique()):
            yield y, Z[Z["연도"] != y], Z[Z["연도"] == y]
        return
    if scheme == "앞으로 가는 검증":
        cuts = [(y, Z["연도"] < y, Z["연도"] == y) for y in C.FORWARD_YEARS]
    else:
        cuts = [("2022~2025", Z["연도"] < min(C.HOLDOUT_YEARS), Z["연도"].isin(C.HOLDOUT_YEARS))]
    for name, mtr, mte in cuts:
        tr = Z[mtr]
        cnt = tr.groupby("조합").size()
        keep = cnt[cnt >= C.FORWARD_MIN_TRAIN].index
        yield name, tr[tr["조합"].isin(keep)], Z[mte & Z["조합"].isin(keep)]


# 검증 방식 하나를 돌린다
# 줄 단위 예측(긴 표)과 검증 한 번마다의 기울기를 돌려준다
def run(Z, scheme, specs, species=()):
    parts, slopes = [], []
    for name, tr, te in folds(Z, scheme):
        if te.empty or tr.empty:
            continue
        assert not set(tr["연도"]) & set(te["연도"]), "학습과 검증에 같은 해가 들어갔다"
        preds, sl = predict_fold(tr, te, specs, species)
        for sid, (p, h) in preds.items():
            d = te[ROW].copy()
            d["spec"], d["pred"], d["half"], d["fold"] = sid, p.to_numpy(), h.to_numpy(), name
            parts.append(d)
        slopes += [{"fold": name, "spec": k, "기울기": v} for k, v in sl.items()]
    P = pd.concat(parts, ignore_index=True)
    P["검증방식"] = scheme
    return P, pd.DataFrame(slopes)


# 누출 점검이다
# 누출은 검증할 해의 정보가 학습에 새어 드는 것이다
# 검증 줄의 실제 날짜를 999로 바꿔도 예측이 그대로인지 본다
# 어긋난 줄 수를 돌려준다
def leak_test_target(Z, scheme, specs, species=()):
    bad = 0
    for _, tr, te in folds(Z, scheme):
        if te.empty or tr.empty:
            continue
        a, _ = predict_fold(tr, te, specs, species)
        b, _ = predict_fold(tr, te.assign(y=999.0, 날짜숫자=999.0), specs, species)
        for k in a:
            pa, pb = a[k][0].to_numpy(dtype=float), b[k][0].to_numpy(dtype=float)
            bad += int((~np.isclose(pa, pb, equal_nan=True)).sum())
    return bad


# 성능 지표

# 모델 하나(sid)를 기준 모델과 같은 줄에서 견준다
# 요약 dict와 해별 표를 돌려준다
# ref가 날짜 기준일이면 그날까지 이미 핀 줄은 뺀다
def evaluate(P, sid, base_sid, ref=None, mask=None):
    a = P[P["spec"] == sid].set_index(["조합", "연도"])
    b = P[P["spec"] == base_sid].set_index(["조합", "연도"])[["pred", "half"]]
    j = a.join(b, rsuffix="_기준", how="inner").dropna(subset=["pred", "pred_기준", "y"])
    if mask is not None:
        j = j[mask(j)]
    excl = 0
    if ref in C.REF_DATES and j["날짜숫자"].notna().any():
        doy = np.array([F.ref_doy(ref, y) for y in j.index.get_level_values("연도")])
        early = j["날짜숫자"].to_numpy() <= doy
        excl, j = int(early.sum()), j[~early]
    if j.empty:
        return None, None
    e, e0 = j["pred"] - j["y"], j["pred_기준"] - j["y"]
    yrs = j.index.get_level_values("연도")
    by = pd.DataFrame({"a1": e.abs().to_numpy(), "a0": e0.abs().to_numpy(), "e": e.to_numpy(), "연도": yrs})
    by = by.groupby("연도").agg(s1=("a1", "sum"), s0=("a0", "sum"), 편향=("e", "mean"), 줄수=("a1", "size"))
    res = {"줄수": len(j), "해수": int(yrs.nunique()), "제외줄수": excl,
           "MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())), "편향": e.mean(),
           "7일이내_비율": float((e.abs() <= C.CLOSE_DAYS).mean()),
           "기준대비_MAE차": np.nan, "MAE차_95하한": np.nan, "MAE차_95상한": np.nan, "이긴_해수": np.nan,
           "구간80_포함률": np.nan}
    ok = j["half"].notna().to_numpy()
    if ok.any():
        res["구간80_포함률"] = float((e.abs().to_numpy()[ok] <= j["half"].to_numpy()[ok]).mean())
    if sid != base_sid:
        res["기준대비_MAE차"] = e.abs().mean() - e0.abs().mean()
        res["이긴_해수"] = int((by["s1"] < by["s0"]).sum())
        if len(by) >= 2:
            rng = np.random.default_rng(C.SEED)
            idx = rng.integers(0, len(by), (C.BOOT_N, len(by)))
            s1, s0, n = (by[c].to_numpy() for c in ("s1", "s0", "줄수"))
            boot = (s1[idx].sum(1) - s0[idx].sum(1)) / n[idx].sum(1)
            res["MAE차_95하한"], res["MAE차_95상한"] = np.percentile(boot, [2.5, 97.5])
    yr = by.assign(MAE=by["s1"] / by["줄수"]).reset_index()[["연도", "줄수", "MAE", "편향"]]
    return res, yr


# 5절 4항 규칙으로 채택할지 정한다
# 두 조건을 다 채우면 채택한다
# 기준 모델보다 MAE가 하루의 절반 이상 작아야 한다
# 이긴 해 수가 정해 둔 수 이상이어야 한다
def adopt(res, wins_needed):
    if res is None or pd.isna(res["기준대비_MAE차"]):
        return False
    return bool(res["기준대비_MAE차"] <= -C.RULE_MAE and res["이긴_해수"] >= wins_needed)


# 비교표 한 줄의 정의를 만든다
# 모델 s를 기준 모델 base와 같은 줄에서 견준다
# ref가 날짜 기준일이면 그날까지 핀 줄을 뺀다
def item(s, base, ref=None):
    return {"sid": s["id"], "base": base["id"], "ref": ref, "기준일": s["기준일"], "모델": s["모델"], "입력": s["입력"],
            "기준모델": base["모델"] + (f"({base['입력']})" if base["입력"] else "")}


# 비교표 줄과 해별 줄을 만든다
# groups는 종집단 이름마다 줄을 고르는 함수 또는 None을 적은 묶음이다
def compare(P, items, meta, groups=None):
    rows, yrs = [], []
    for gname, gmask in (groups or {"전체": None}).items():
        for it in items:
            res, yr = evaluate(P, it["sid"], it["base"], ref=it["ref"], mask=gmask)
            if res is None:
                continue
            info = {**meta, "기준일": it["기준일"], "모델": it["모델"], "입력": it["입력"], "기준모델": it["기준모델"],
                    "종집단": gname}
            rows.append({**info, **res})
            yrs.append(yr.assign(**info))
    return rows, yrs


# 나눠 볼 종집단을 돌려준다
# 정제본 봄꽃이면 전체와 이른 종과 늦은 종이다
# 아니면 전체만 돌려준다
def species_groups(Z):
    g = {"전체": None}
    if {"이른 종", "늦은 종"} <= set(Z["종집단"]):
        g["이른 종"] = lambda j: (j["종집단"] == "이른 종").to_numpy()
        g["늦은 종"] = lambda j: (j["종집단"] == "늦은 종").to_numpy()
    return g


COMP_COLS = ["현상", "목표단계", "검증방식", "기준일", "모델", "입력", "기준모델", "종집단", "자료판", "줄수", "해수",
             "제외줄수", "MAE", "RMSE", "편향", "7일이내_비율", "기준대비_MAE차", "MAE차_95하한", "MAE차_95상한",
             "이긴_해수", "구간80_포함률"]
YEAR_COLS = ["현상", "목표단계", "검증방식", "기준일", "모델", "입력", "종집단", "자료판", "연도", "줄수", "MAE", "편향"]


INT_COLS = ("줄수", "해수", "제외줄수", "이긴_해수")
TEXT_COLS = ("입력", "기준모델", "종집단", "자료판")


# 줄 목록이나 표를 정해진 열 순서로 맞추고 반올림한다
# 빈 글자 칸은 빈 문자열로 채운다
# 개수 칸은 정수로 바꾼다
def tidy(rows, cols):
    T = (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).reindex(columns=cols)
    for c in T.columns:
        if c in TEXT_COLS:
            T[c] = T[c].fillna("").astype(str)
        elif c in INT_COLS:
            T[c] = pd.to_numeric(T[c], errors="coerce").round().astype("Int64")
        elif T[c].dtype.kind == "f":
            T[c] = T[c].round(3 if c.endswith(("비율", "포함률")) else 2)
    return T


def models_dir():
    d = C.out_dir("work") / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
