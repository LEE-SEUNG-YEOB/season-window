# 9단계 지난 해마다 그해 앞 기록만으로 기능 2를 다시 돌려 실제와 견줌

import datetime as dt
import functools
import math
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import mconfig as C
import models as M

COMBO = ["수목원", "종"]
INPUTS = ("history.csv", "bloom_pairs.csv", "combo_stats.csv", "gaps.csv", "shift.csv", "decision.csv")
COLS = COMBO + ["연도", "앞해수", "평균", "표준편차", "최근5년_편차", "길이", "짝해수"]
DAYS = 9 # 판정에 쓰는 축제 일수(화면 기본값)
LONGEST = 30 # 필요 일수를 찾는 가장 긴 일수
NEEDS = (1, 3, 7) # 겹침 기준 일수
JUDGED = (1, 7) # 판정하는 겹침 기준
SPAN = 45 # 가장 좋은 기간을 찾는 범위(평균 앞뒤 일수)
OVER = 0.10 # 과신으로 보는 차이
TARGET = 0.8 # 필요 일수의 기준 비율(등급 높음)
GAIN = 0.5 # 권장 판정에 쓰는 꽃 핀 날 수 차이
SMALL = 20 # 종별 필요 일수를 내는 가장 적은 줄 수
NOW, PLAIN = "지금 규칙", "편차 없음"
MAIN, WIDE = "앞 해 8해 이상", "앞 해 3해 이상"
RATES = ["확률_평균", "실제_비율", "차이", "차이_95하한", "차이_95상한", "실제_95하한", "실제_95상한",
         "세기_평균", "세기_실제", "세기_차이", "세기_95하한", "세기_95상한"]
SPANS = ["꽃핀날_최적", "꽃핀날_작년", "꽃핀날_차이", "꽃핀날_95하한", "꽃핀날_95상한"]

def table(name):
    return pd.read_csv(C.SERVICE / name, encoding="utf-8-sig")

# 대상 해보다 앞선 해만으로 조합 통계와 꽃 시기 길이와 최근 5년 편차 계산
# 6단계가 서비스 표를 만드는 것과 같은 정의
def prior(H, B, Z, y):
    h = H[(H["현상"] == C.STAGE_START) & (H["연도"] < y)]
    S = h.groupby(COMBO)["날짜숫자"].agg(앞해수="count", 평균="mean", 표준편차="std").reset_index()
    S = S.round({"평균": 3, "표준편차": 3})
    b = B[B["연도"] < y]
    own = b.groupby(COMBO)["개화기간"].agg(조합길이="median", 짝해수="count").reset_index()
    kind = b.groupby("종")["개화기간"].median().rename("종길이").reset_index()
    S = S.merge(own, on=COMBO, how="left").merge(kind, on="종", how="left")
    S["짝해수"] = S["짝해수"].fillna(0).astype(int)
    # 짝 기록이 8해 이상이면 조합 중앙값 아니면 종 중앙값
    S["길이"] = S["조합길이"].where(S["짝해수"] >= C.MIN_YEARS, S["종길이"])
    z = Z[(Z["현상"] == "개화") & (Z["연도"] < y)]
    S["최근5년_편차"] = round(M.drift(z), 2)
    S.insert(2, "연도", y)
    return S

# 모든 해를 넣어 다시 낸 값이 6단계 서비스 표와 같은지 확인
# 다른 곳을 글로 반환
def faults(H, B, Z):
    S = prior(H, B, Z, max(C.YEARS) + 1)
    V = table("combo_stats.csv")
    J = S.merge(V[V["현상"] == C.STAGE_START], on=COMBO, how="outer", indicator=True)
    same = (J["앞해수"] == J["관측연수"]) & (J["평균_x"] == J["평균_y"]) \
        & ((J["표준편차_x"] == J["표준편차_y"]) | (J["표준편차_x"].isna() & J["표준편차_y"].isna()))
    bad = []
    if (J["_merge"] != "both").any() or not same.all():
        bad.append(f"조합 통계가 서비스 표와 다름({int((~same).sum())}개)")
    G = table("gaps.csv")
    own, kind = G[G["구분"] == "조합"], G[G["구분"] == "종"].set_index("종")["기간_중앙값"]
    L = S.merge(own[COMBO + ["기간_중앙값", "기간_해수"]], on=COMBO, how="left")
    want = L["기간_중앙값"].where(L["기간_해수"] >= C.MIN_YEARS, L["종"].map(kind))
    if not ((L["길이"] == want) | (L["길이"].isna() & want.isna())).all():
        bad.append("꽃 시기 길이가 서비스 표와 다름")
    D = table("shift.csv")
    if S["최근5년_편차"].iloc[0] != float(D.loc[D["현상"] == "개화", "최근5년_편차"].iloc[0]):
        bad.append("최근 5년 편차가 서비스 표와 다름")
    return bad

# 대상 해마다 그해 앞 값을 붙인 표 생성
# 그해에 개화시작과 낙화가 함께 있는 조합만 남김
# 계산할 수 없어 뺀 줄 수도 반환
def sheet(H, B, Z):
    rows = [prior(H, B, Z, y).merge(B.loc[B["연도"] == y, COMBO], on=COMBO) for y in C.FORWARD_YEARS]
    T = pd.concat(rows, ignore_index=True)
    T = T[T["앞해수"] >= C.FORWARD_MIN_TRAIN]
    ok = T["표준편차"].notna() & (T["표준편차"] > 0) & T["길이"].notna()
    return T.loc[ok, COLS].sort_values(COMBO + ["연도"]).reset_index(drop=True), int((~ok).sum())

# 최근 5년 보정이 개화에 채택됐는지 확인
def adoption():
    D = table("decision.csv")
    r = D[(D["항목"] == "M0r") & (D["현상"] == "개화")]
    return bool(len(r)) and r["판정"].iloc[0] == "채택"

# 어느 해의 달과 날을 날짜숫자로 변환
# 2월 29일이 없는 해는 2월 28일 사용
@functools.lru_cache(maxsize=None)
def yearday(year, month, day):
    try:
        return dt.date(year, month, day).timetuple().tm_yday
    except ValueError:
        return dt.date(year, 2, 28).timetuple().tm_yday

# 한 해의 날짜숫자를 다른 해의 같은 달과 날로 변환
def sameday(doy, year, target):
    d = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    return yearday(target, d.month, d.day)

# 기능 2와 같은 정규 누적확률
def cdf(x, mu, sd):
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))

# 시작일 후보마다 쓰는 위쪽과 아래쪽 누적확률을 미리 계산
# 식과 계산 순서는 기능 2와 같음
def curve(mu, sd, length):
    first, last = max(1, int(mu) - SPAN), int(mu) + SPAN
    ks = range(first, last + LONGEST + 1)
    up = np.array([cdf(k + 0.5, mu, sd) for k in ks])
    low = np.array([cdf(k - length - 0.5, mu, sd) for k in ks])
    return np.arange(last - first + 1), first, up, low

# 확률이 가장 높은 시작일과 그 확률 반환
# 값이 같으면 가장 이른 시작일(기능 2와 같은 순서)
def peak(cv, d, m):
    a, first, up, low = cv
    p = np.maximum(0.0, up[a + d - m] - low[a + m - 1])
    k = int(np.argmax(p))
    return first + k, float(p[k])

# 기간이 꽃 시기와 겹친 날 수
def bloom(s, d, on, off):
    lo, hi = max(s, math.ceil(on)), min(s + d - 1, math.floor(off))
    return max(0, hi - lo + 1)

# 가장 가까운 앞 해의 꽃 시기 한가운데에 맞춘 기간이 그해 꽃과 겹친 날 수
# 앞 해에 짝 기록이 없으면 빈 값
def rival(own, prev, y, d):
    if not prev:
        return np.nan
    on, off = own[prev[-1]]
    s = math.floor((on + off) / 2 - (d - 1) / 2 + 0.5)
    return bloom(sameday(s, prev[-1], y), d, *own[y])

# 줄과 방식과 일수와 겹침 기준마다 가장 좋은 기간이 말한 값과 실제 계산
# 말한 값은 기능 함수처럼 소수 셋째 자리로 반올림
def cells(T, B, use):
    pairs = {}
    for s, k, y, on, off in zip(B["수목원"], B["종"], B["연도"].tolist(), B["개화시작"].tolist(), B["낙화"].tolist()):
        pairs.setdefault((s, k), {})[y] = (on, off)
    out = []
    for r in T.to_dict("records"):
        y, n = int(r["연도"]), int(r["앞해수"])
        own = pairs[(r["수목원"], r["종"])]
        on, off = own[y]
        prev = sorted(k for k in own if k < y)
        sd, length = float(r["표준편차"]) * math.sqrt(1 + 1 / n), float(r["길이"])
        modes = [(NOW, float(r["평균"]) + (float(r["최근5년_편차"]) if use else 0.0), range(1, LONGEST + 1))]
        if use:
            modes.append((PLAIN, float(r["평균"]), (DAYS,)))
        for mode, mu, span in modes:
            cv = curve(mu, sd, length)
            for d in span:
                last = rival(own, prev, y, d)
                for m in NEEDS:
                    if d < m:
                        continue
                    s, p = peak(cv, d, m)
                    got = bloom(s, d, on, off)
                    # 지난 해 세기는 달과 날을 맞춰 옮긴 기간과 그해 실제 낙화로 계산
                    hit = sum(bloom(sameday(s, y, k), d, *own[k]) >= m for k in prev)
                    count = round(hit / len(prev), 3) if len(prev) >= C.MIN_YEARS else np.nan
                    out.append((r["수목원"], r["종"], y, n, mode, d, m, round(p, 3), int(got >= m), count, got, last))
    L = pd.DataFrame(out, columns=COMBO + ["연도", "앞해수", "방식", "일수", "겹침기준", "확률", "겹침", "세기", "꽃핀날", "작년꽃핀날"])
    L["종집단"] = np.where(L["종"].isin(C.EARLY_SPECIES), "이른 종", np.where(L["종"].isin(C.LATE_SPECIES), "늦은 종", "그 밖"))
    return L

# 해 묶음 부트스트랩으로 비율의 95% 구간 계산
# 해가 둘 미만이면 빈 값
def interval(by, top):
    if len(by) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, len(by), (C.BOOT_N, len(by)))
    a, n = by[top].to_numpy(), by["줄수"].to_numpy()
    lo, hi = np.percentile(a[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    return float(lo), float(hi)

# 한 칸에 드는 줄을 묶어 말한 값과 실제와 꽃 핀 날 수 계산
def pool(g):
    by = g.groupby("연도").agg(줄수=("겹침", "size"), 확률=("확률", "sum"), 겹침=("겹침", "sum"))
    by["차이"] = by["확률"] - by["겹침"]
    n = int(by["줄수"].sum())
    res = {"줄수": n, "해수": len(by), "조합수": int(g.groupby(COMBO).ngroups),
           "확률_평균": by["확률"].sum() / n, "실제_비율": by["겹침"].sum() / n}
    res["차이"] = res["확률_평균"] - res["실제_비율"]
    res["차이_95하한"], res["차이_95상한"] = interval(by, "차이")
    res["실제_95하한"], res["실제_95상한"] = interval(by, "겹침")
    # 지난 해 세기는 앞선 짝 기록이 8해 이상인 줄만
    c = g[g["세기"].notna()]
    cy = c.groupby("연도").agg(줄수=("겹침", "size"), 세기=("세기", "sum"), 겹침=("겹침", "sum"))
    cy["차이"] = cy["세기"] - cy["겹침"]
    k = int(cy["줄수"].sum())
    res["세기_줄수"], res["세기_해수"] = k, len(cy)
    res["세기_평균"], res["세기_실제"] = (cy["세기"].sum() / k, cy["겹침"].sum() / k) if k else (np.nan, np.nan)
    res["세기_차이"] = res["세기_평균"] - res["세기_실제"]
    res["세기_95하한"], res["세기_95상한"] = interval(cy, "차이")
    # 꽃 핀 날 수는 작년 기간을 낼 수 있는 줄만
    v = g[g["작년꽃핀날"].notna()]
    vy = v.groupby("연도").agg(줄수=("꽃핀날", "size"), 최적=("꽃핀날", "sum"), 작년=("작년꽃핀날", "sum"))
    vy["차이"] = vy["최적"] - vy["작년"]
    k = int(vy["줄수"].sum())
    res["견준_줄수"], res["견준_해수"] = k, len(vy)
    res["꽃핀날_최적"], res["꽃핀날_작년"] = (vy["최적"].sum() / k, vy["작년"].sum() / k) if k else (np.nan, np.nan)
    res["꽃핀날_차이"] = res["꽃핀날_최적"] - res["꽃핀날_작년"]
    res["꽃핀날_95하한"], res["꽃핀날_95상한"] = interval(vy, "차이")
    # 한 해의 승패는 그해 줄들의 평균이고 같으면 이긴 해가 아님
    res["이긴_해수"] = int((vy["최적"] > vy["작년"]).sum())
    return res

# 틀과 방식과 묶음마다 칸을 모은 표 생성
# 판단 틀의 지금 규칙만 모든 일수를 싣고 나머지는 판정에 쓰는 일수만 실음
def grid(L):
    rows = []
    for frame, least in ((MAIN, C.MIN_YEARS), (WIDE, C.FORWARD_MIN_TRAIN)):
        for mode, g0 in L[L["앞해수"] >= least].groupby("방식", sort=False):
            for kind, key in (("전체", None), ("종집단", "종집단"), ("종", "종"), ("해", "연도")):
                full = frame == MAIN and mode == NOW and kind != "해"
                g1 = g0 if full else g0[g0["일수"] == DAYS]
                for k, g in g1.groupby(([key] if key else []) + ["일수", "겹침기준"], sort=True):
                    rows.append({"틀": frame, "방식": mode, "묶음": kind, "이름": str(k[0]) if key else "전체",
                                 "일수": int(k[-2]), "겹침기준": int(k[-1]), **pool(g)})
    return pd.DataFrame(rows)

# 말한 값과 실제의 차이로 과신 여부 판정
def label(diff, lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return "판정 못 함"
    if diff > OVER and lo > 0:
        return "과신"
    if diff < -OVER and hi < 0:
        return "낮춰 말함"
    return "과신의 증거 없음"

# 판단 칸의 판정표 생성
# 기준은 결과를 보기 전에 정한 값
def verdicts(G):
    g = G[(G["틀"] == MAIN) & (G["방식"] == NOW) & (G["묶음"] == "전체") & (G["일수"] == DAYS)].set_index("겹침기준")
    head = {"틀": MAIN, "방식": NOW, "일수": DAYS}
    rule = f"말한 값이 실제보다 {OVER:.2f} 넘게 높고 해 묶음 95% 구간이 0을 벗어나면 과신"
    rows = []
    for item, pre, n, years in (("식의 확률 대 실제 겹친 비율", "", "줄수", "해수"),
                                ("지난 해 세기 대 실제 겹친 비율", "세기_", "세기_줄수", "세기_해수")):
        for m in JUDGED:
            r = g.loc[m]
            a, b = (r["확률_평균"], r["실제_비율"]) if not pre else (r["세기_평균"], r["세기_실제"])
            lo, hi = (r["차이_95하한"], r["차이_95상한"]) if not pre else (r["세기_95하한"], r["세기_95상한"])
            rows.append({"항목": item, **head, "겹침기준": m, "줄수": int(r[n]), "해수": int(r[years]), "값": a, "견준_값": b,
                         "차이": a - b, "차이_95하한": lo, "차이_95상한": hi, "이긴_해수": pd.NA, "규칙": rule,
                         "판정": label(a - b, lo, hi)})
    # 권장 판정은 겹침 기준 하루 이상으로 고른 기간 사용
    r = g.loc[JUDGED[0]]
    years = int(r["견준_해수"])
    need = max(years - 1, 1)
    ok = years > 0 and r["꽃핀날_차이"] >= GAIN and r["이긴_해수"] >= need
    rows.append({"항목": "권한 기간 대 작년 꽃 시기에 맞춘 기간(꽃 핀 날 수)", **head, "겹침기준": JUDGED[0],
                 "줄수": int(r["견준_줄수"]), "해수": years, "값": r["꽃핀날_최적"], "견준_값": r["꽃핀날_작년"],
                 "차이": r["꽃핀날_차이"], "차이_95하한": r["꽃핀날_95하한"], "차이_95상한": r["꽃핀날_95상한"],
                 "이긴_해수": int(r["이긴_해수"]), "규칙": f"평균 {GAIN}일 이상 많고 {years}해 중 {need}해 이상",
                 "판정": "작년에 맞추는 것보다 낫다" if ok else "차이를 확인하지 못함"})
    D = pd.DataFrame(rows)
    D["이긴_해수"] = D["이긴_해수"].astype("Int64")
    return D

# 그 일수부터 가장 긴 일수까지 줄곧 기준 비율 이상인 가장 짧은 일수
# 닿지 않으면 빈 값
def reach(q, col):
    best = pd.NA
    for d, v in zip(q["일수"].to_numpy()[::-1], q[col].to_numpy()[::-1]):
        if not v >= TARGET:
            break
        best = int(d)
    return best

# 전체와 종집단과 종별 필요 일수 표 생성
# 판정 없이 싣기만 하는 값
def needs(G):
    g = G[(G["틀"] == MAIN) & (G["방식"] == NOW) & G["묶음"].isin(["전체", "종집단", "종"])]
    rows = []
    for (kind, name, m), q in g.groupby(["묶음", "이름", "겹침기준"], sort=False):
        q = q.sort_values("일수")
        n, at = int(q["줄수"].iloc[0]), q.set_index("일수")["실제_비율"]
        row = {"묶음": kind, "이름": name, "겹침기준": int(m), "줄수": n, "해수": int(q["해수"].iloc[0]),
               "비율_9일": at.get(DAYS, np.nan), "비율_30일": at.get(LONGEST, np.nan),
               "필요일수": reach(q, "실제_비율"), "필요일수_하한기준": reach(q, "실제_95하한") if kind != "종" else pd.NA, "비고": ""}
        if kind == "종" and n < SMALL:
            row["필요일수"], row["비고"] = pd.NA, f"줄 수 {SMALL} 미만이라 내지 않음"
        elif pd.isna(row["필요일수"]):
            row["비고"] = f"{LONGEST}일 안에 닿지 않음"
        rows.append(row)
    N = pd.DataFrame(rows)
    for c in ("필요일수", "필요일수_하한기준"):
        N[c] = N[c].astype("Int64")
    return N.round({"비율_9일": 3, "비율_30일": 3})

def main():
    if C.mode() != "정제본":
        print("정제본(data/interim)이 없어 되돌려 보기 표를 만들지 않는다.")
        return
    miss = [f for f in INPUTS if not (C.SERVICE / f).exists()]
    if miss:
        sys.exit(f"!! 6단계 출력이 없음 {miss}")
    H, B = table("history.csv"), table("bloom_pairs.csv")
    Z = M.dataset(C.folder("features") / C.FEATURES.name)
    bad = faults(H, B, Z)
    if bad:
        sys.exit("!! 되돌려 보기 점검 실패: " + " / ".join(bad))
    T, dropped = sheet(H, B, Z)
    out = C.SERVICE / "hindcast.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out.relative_to(C.ROOT)} {T.shape}")
    # 기능 함수와 같은 숫자를 쓰도록 저장한 표를 다시 읽어 계산
    R = table("hindcast.csv")
    if R.shape != T.shape or not (R[COLS] == T[COLS]).all().all() or R.duplicated(COMBO + ["연도"]).any():
        sys.exit("!! 되돌려 보기 점검 실패: 저장한 표를 다시 읽은 값이 다르거나 같은 줄이 두 번")
    use = adoption()
    G = grid(cells(R, B, use))
    D, N = verdicts(G), needs(G)
    tdir = C.folder("tables")
    for name, X in (("model_hindcast.csv", G.round({**{c: 3 for c in RATES}, **{c: 2 for c in SPANS}})),
                    ("model_hindcast_decision.csv", D.round({"값": 3, "견준_값": 3, "차이": 3, "차이_95하한": 3, "차이_95상한": 3})),
                    ("model_hindcast_days.csv", N)):
        X.to_csv(tdir / name, index=False, encoding="utf-8-sig")
        print(f"저장: {(tdir / name).relative_to(C.ROOT)} {X.shape}")
    core = int((R["앞해수"] >= C.MIN_YEARS).sum())
    print(f"판단 틀({MAIN}) {core}줄 참고 틀({WIDE}) {len(R)}줄 계산할 수 없어 뺀 줄 {dropped} 최근 5년 편차 사용 {use}")
    print(D.round(3).to_string(index=False))
    print(N.to_string(index=False))
    print("해가 9개 안팎이라 해 묶음 구간은 실제보다 좁게 나올 수 있다")

if __name__ == "__main__":
    main()
