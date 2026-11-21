# 시계 모듈 적산온도 시계와 그해 관측 신호와 잘린 정규 예측

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtri

import features as F
import mconfig as C

DAYS = 367 # 날짜숫자 칸 수(0번 칸은 비움)
HALFLOG = 0.5 * np.log(2 * np.pi)
KEY = ["수목원", "종", "연도"]
SIGNALS = ["시계", "전국", "수목원", "꽃눈파열", "개화시작"]
STATES = ["앞 단계 없음", "꽃눈파열 봄", "개화시작 봄"]

# 사건 표 읽기
# 사건은 한 수목원 한 종 한 단계가 한 해에 관측된 날짜 한 줄
# 계열은 수목원과 종과 단계가 같은 사건 묶음
# 관측 해가 모자란 계열과 이상치 의심 줄 제외
def events():
    S = pd.read_csv(C.PHENO_SITE)
    S = S[S["연도"].isin(C.YEARS)]
    key = ["수목원", "종", "현상"]
    if S.duplicated(key + ["연도"]).any():
        raise ValueError("사건 표에 같은 계열 같은 해가 두 번 있다")
    n = S.groupby(key)["연도"].transform("count")
    E = S[(n >= C.MIN_YEARS) & ~S["이상치의심"].astype(bool)]
    return E[key + ["연도", "날짜숫자"]].reset_index(drop=True)

# 사건과 목표 줄의 번호 표 생성
# B는 학습 표의 봄꽃 줄
# 목표 줄마다 제 개화 사건과 같은 나무의 꽃눈파열 사건과 개화시작 사건 번호 연결(없으면 -1)
def layout(E, B):
    E = E.reset_index(drop=True)
    sites = sorted(E["수목원"].unique())

    def pick(stage):
        q = E[E["현상"] == stage]
        at = pd.Series(q.index, index=pd.MultiIndex.from_frame(q[KEY]))
        return at.reindex(pd.MultiIndex.from_frame(B[KEY])).fillna(-1).astype(int).to_numpy()

    L = {"sites": sites,
         "site": E["수목원"].map({s: i for i, s in enumerate(sites)}).to_numpy(),
         "year": (E["연도"] - C.YEARS[0]).to_numpy(),
         "series": pd.factorize(E["수목원"] + "|" + E["종"] + "|" + E["현상"])[0],
         "group": pd.factorize(E["수목원"] + "|" + E["종"])[0],
         "date": E["날짜숫자"].to_numpy(float),
         "target": pick(C.STAGE_SPRING), "bud": pick(C.STAGE_BUD), "start": pick(C.STAGE_START),
         "combo": pd.factorize(B["조합"])[0]}
    if (L["target"] < 0).any():
        raise ValueError("목표 줄이 사건 표에 없다")
    if not np.array_equal(L["date"][L["target"]], B["y"].to_numpy(float)):
        raise ValueError("목표 줄 날짜가 사건 표와 다르다")
    return L

# 수목원 x 해 x 날짜숫자 하루 평균기온 표 생성
# 날짜숫자는 1월 1일이 1
def temps(daily, sites):
    st = C.stations().set_index("수목원")["기상지점"]
    W = np.full((len(sites), len(C.YEARS), DAYS), np.nan)
    for i, s in enumerate(sites):
        g = F.frame(daily, st[s], f"{C.YEARS[0]}-01-01", f"{C.YEARS[-1]}-12-31")[F.TEMP]
        for k, y in enumerate(C.YEARS):
            v = g[g.index.year == y].to_numpy(float)
            W[i, k, 1:len(v) + 1] = v
    return W

# 누적온도 표 생성
# 시계 시작일부터 하루 평균기온에서 기준 온도를 뺀 값을 쌓음(음수와 빠진 날은 0)
# sum은 수목원 x 해 x 날짜숫자 누적값
# ref는 기준일별 해마다의 날짜숫자
# tail은 기준일별 그날 뒤로 늘어난 양
def heat(W, refs):
    g = np.nan_to_num(np.clip(W - C.GDD_BASE, 0, None))
    g[:, :, :C.M2_START] = 0
    SM = np.cumsum(g, axis=2)
    rt = {ref: np.array([F.refdoy(ref, y) for y in C.YEARS]) for ref in refs}
    tail = {}
    for ref in refs:
        span = C.M2_CAP - int(rt[ref].min()) + 1
        inc = np.zeros(SM.shape[:2] + (span,))
        for t, r in enumerate(rt[ref]):
            inc[:, t, :] = SM[:, t, r:r + span] - SM[:, t, r:r + 1]
        tail[ref] = inc
    return {"sum": SM, "ref": rt, "tail": tail}

# 기준일에 본 시계 날짜 계산
# quota는 사건마다 그 계열이 그 단계에 닿을 때 쌓여 있던 누적온도의 학습 해 평균
# 기준일에 이미 닿았으면 닿은 날 사용
# 못 닿았으면 학습 해마다 기준일 뒤로 늘어난 양을 이어 붙여 닿는 날을 구하고 그 중앙값 사용
# 그 사건의 해는 이어 붙이는 해에서 제외
# 누적온도가 요구량과 같은 날은 닿은 날
# 소수 끝자리 오차로 같은 값이 모자란 값으로 읽히지 않게 요구량을 아주 조금 낮춰 비교
def clock(L, H, quota, ref, oky):
    es, et = L["site"], L["year"]
    SM, inc = H["sum"], H["tail"][ref]
    r = H["ref"][ref][et]
    quota = quota - C.M2_TIE
    need = quota - SM[es, et, r]
    x = np.empty(len(es))
    ny = len(oky)
    for s in range(SM.shape[0]):
        here = es == s
        for t in range(ny):
            m = here & (et == t) & (need <= 0)
            if m.any():
                x[m] = np.maximum(np.searchsorted(SM[s, t], quota[m], side="left"), C.M2_START)
        u = np.flatnonzero(here & (need > 0))
        if not len(u):
            continue
        k = np.column_stack([np.searchsorted(inc[s, t], need[u], side="left") for t in range(ny)])
        xx = np.minimum(r[u][:, None] + k, C.M2_CAP).astype(float)
        row = np.arange(len(u))
        xx[row, et[u]] = np.inf
        xx[:, ~oky] = np.inf
        xx.sort(axis=1)
        nv = int(oky.sum()) - oky[et[u]]
        x[u] = 0.5 * (xx[row, (nv - 1) // 2] + xx[row, nv // 2])
    return x

# 아직 오지 않은 사건이 평년보다 늦을 것으로 기대되는 정도 계산
# 표준 정규 분포에서 z보다 클 때의 평균
def hazard(z):
    z = np.minimum(z, 30.0)
    return np.exp(-0.5 * z * z - HALFLOG - log_ndtr(-z))

# 아래를 lo에서 자른 정규 분포의 분위 계산
def quantile(eta, sig, lo, p):
    al = np.minimum((lo - eta) / sig, 30.0)
    u = np.exp(np.log1p(-p) + log_ndtr(-al))
    return np.maximum(eta - sig * ndtri(u), lo)

# 잘린 정규 가능도로 신호별 이득과 상태별 폭 배수를 함께 추정
# 값이 0이 아닌 줄이 모자란 신호의 이득은 0으로 고정
# 줄이 모자란 상태는 앞 단계 없음 상태의 폭 배수 사용
# 이득과 폭 배수와 추정이 끝났는지 반환
def gains(Z, y, mu, s, lo, st):
    nb = Z.shape[1]
    act = (Z != 0).sum(axis=0) >= C.M2_ROWS
    cn = np.bincount(st, minlength=3)
    smap = np.array([0, 1 if cn[1] >= C.M2_ROWS else 0, 2 if cn[2] >= C.M2_ROWS else 0])
    q = smap[st]

    def loss(th):
        sig = s * np.exp(th[nb:])[q]
        eta = mu + Z @ th[:nb]
        z = (y - eta) / sig
        al = np.minimum((lo - eta) / sig, 30.0)
        ls = log_ndtr(-al)
        h = np.exp(-0.5 * al * al - HALFLOG - ls)
        val = float(np.sum(np.log(sig) + 0.5 * z * z + ls))
        grad = np.concatenate([Z.T @ ((h - z) / sig), np.bincount(q, 1 - z * z + h * al, 3)])
        return val, grad

    wide = tuple(np.log(C.M2_WIDTH))
    bd = [C.M2_GAIN if a else (0.0, 0.0) for a in act]
    bd += [wide if (k == 0 or smap[k] == k) else (0.0, 0.0) for k in range(3)]
    res = minimize(loss, np.zeros(nb + 3), jac=True, method="L-BFGS-B", bounds=bd)
    return res.x[:nb], np.exp(res.x[nb:])[smap], bool(res.success)

# 검증 한 번의 한 기준일 예측 계산
# d는 사건 날짜이고 oky는 학습 해 표시이고 trb는 목표 줄 가운데 학습 줄 표시
# lag는 전국 최근 5년 편차(주면 기준일에 자른 M0r도 계산)
# 목표 줄 전체 길이로 모델별 예측(가운데와 아래와 위)과 추정값 반환
# 학습 줄에는 제 해를 뺀 값이 들어가므로 검증 줄 값만 사용
def fold(L, H, d, oky, trb, ref, lag=None):
    es, et, ej, eg, ib, cb = (L[k] for k in ("site", "year", "series", "group", "target", "combo"))
    NS, NY = H["sum"].shape[:2]
    NJ, NG, NC = ej.max() + 1, eg.max() + 1, cb.max() + 1
    sb, gb, tb, yb = es[ib], eg[ib], et[ib], d[ib]

    # 계열별 학습 해 통계(평년 날짜와 표준편차와 누적온도 요구량)
    # 날짜에 반일이 있으면 내림한 날의 누적온도 사용
    tr = oky[et]
    nj = np.bincount(ej[tr], minlength=NJ).astype(float)
    nn = np.maximum(nj, 1)
    mj = np.bincount(ej[tr], d[tr], NJ) / nn
    sdj = np.sqrt(np.maximum(np.bincount(ej[tr], d[tr] ** 2, NJ) - nn * mj ** 2, 0) / np.maximum(nj - 1, 1))
    sdj = np.maximum(sdj, 1.0)
    okj = nj >= C.M2_SERIES
    hd = H["sum"][es, et, np.clip(d.astype(int), 0, DAYS - 1)]
    quota = np.bincount(ej[tr], hd[tr], NJ) / nn

    # 평년보다 늦은 날 수(학습 사건은 제 해를 뺀 평년과 비교)
    lf = np.where(tr, nj[ej] / np.maximum(nj[ej] - 1, 1), 1.0)
    a = (d - mj[ej]) * lf

    # 조합별 학습 줄 통계(평년 개화일과 기본 폭)
    nc = np.bincount(cb[trb], minlength=NC).astype(float)
    sc = np.bincount(cb[trb], yb[trb], NC)
    muc = sc / np.maximum(nc, 1)
    sdc = np.sqrt(np.maximum(np.bincount(cb[trb], yb[trb] ** 2, NC) - nc * muc ** 2, 0) / np.maximum(nc - 1, 1))
    s = np.maximum(sdc * np.sqrt(1 + 1 / np.maximum(nc, 1)), 1.0)[cb]
    mu = np.where(trb, (sc[cb] - yb) / np.maximum(nc[cb] - 1, 1), muc[cb])

    # 시계 날짜가 평년 시계 날짜보다 늦은 날 수
    rt = H["ref"][ref]
    re = rt[et].astype(float)
    x = clock(L, H, quota[ej], ref, oky)
    xa = (x - (np.bincount(ej[tr], x[tr], NJ) / nn)[ej]) * lf

    # 기준일까지 온 사건은 평년보다 늦은 날 수 사용
    # 아직 안 온 이른 계열 사건은 늦을 것으로 기대되는 날 수 사용
    # 여기서 시계가 설명하는 몫을 뺀 값이 시계가 못 맞춘 몫
    occ = d <= re
    early = (mj[ej] <= re - C.M2_EARLY) & okj[ej]
    wait = sdj[ej] * hazard((re + 0.5 - mj[ej]) / sdj[ej])
    fm = occ & early & tr
    be = float((a[fm] * xa[fm]).sum() / max((xa[fm] ** 2).sum(), 1e-9))
    miss = np.where(occ, a, wait) - be * xa

    # 그해 전국 신호와 같은 수목원 다른 종 신호(이른 계열 사건의 평균을 관측 수에 따라 줄임)
    kst, kgt = es * NY + et, eg * NY + et
    nt = np.bincount(et[early], minlength=NY).astype(float)
    Nt = np.bincount(et[early], miss[early], NY) / np.maximum(nt, 1)
    nst = np.bincount(kst[early], minlength=NS * NY).astype(float)
    sst = np.bincount(kst[early], miss[early], NS * NY)
    ngt = np.bincount(kgt[early], minlength=NG * NY).astype(float)
    sgt = np.bincount(kgt[early], miss[early], NG * NY)
    cn = nst[sb * NY + tb] - ngt[gb * NY + tb]
    cs = sst[sb * NY + tb] - sgt[gb * NY + tb]
    nat = Nt[tb] * nt[tb] / (nt[tb] + C.M2_SHRINK)
    site = np.where(cn > 0, cs / np.maximum(cn, 1) - Nt[tb], 0.0) * cn / (cn + C.M2_SHRINK)

    # 같은 나무의 앞 단계 신호와 기준일까지 봤는지 표시
    def own(evi):
        has = evi >= 0
        e = np.where(has, evi, 0)
        ok = has & okj[ej[e]]
        return np.where(ok & (occ[e] | early[e]), miss[e], 0.0), ok & occ[e]

    iB, sB = own(L["bud"])
    iO, sO = own(L["start"])
    st = np.where(sO, 2, np.where(sB, 1, 0))

    # 기준일에 아직 안 핀 학습 줄로 이득과 폭 배수 추정
    lo = rt[tb] + 0.5
    rem = yb > rt[tb]
    xab = xa[ib]
    Z = np.column_stack([xab, nat, site, iB, iO])
    b1 = float((xab[trb] * (yb[trb] - mu[trb])).sum() / max((xab[trb] ** 2).sum(), 1e-9))
    fr = trb & rem
    beta, rho, done = gains(Z[fr], yb[fr], mu[fr], s[fr], lo[fr], st[fr])

    # 기준일 뒤로 자른 분포의 가운데와 80% 범위
    lev = (1 - C.INTERVAL_LEVEL) / 2

    def band(eta, sig):
        return [quantile(eta, sig, lo, p) for p in (0.5, lev, 1 - lev)]

    pred = {"M0": band(mu, s), "시계만": band(mu + b1 * xab, s), "시계와 관측": band(mu + Z @ beta, s * rho[st])}
    if lag is not None:
        pred["M0r"] = band(mu + lag, s)
    est = {"시계만 기울기": b1, "시계가 못 맞춘 몫 기울기": be}
    est.update({f"{k} 이득": float(v) for k, v in zip(SIGNALS, beta)})
    est.update({f"폭 배수 {k}": float(v) for k, v in zip(STATES, rho)})
    est.update({"학습 줄": int(fr.sum()), "추정 끝": done})
    return {"pred": pred, "base": mu, "state": st, "est": est}
