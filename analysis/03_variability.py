"""종별·지역별 해마다의 변동 폭을 계산한다.

변동 폭 = 수목원·종·단계별로 연도별 날짜숫자의 표준편차(일).
출력 (reports/tables/)
- variability_site_species.csv : 수목원 x 종 x 단계별 관측 연수, 평균, 표준편차 등
- variability_by_species.csv   : 종별 변동 폭 (수목원 평균을 뺀 편차를 모아 계산) + 95% 구간
- variability_by_site.csv      : 수목원별 변동 폭 + 95% 구간
- variability_tests.csv        : 종 간·수목원 간 변동 폭 차이 검정 (순열 검정)
"""
import numpy as np
import pandas as pd
import config as C

MIN_YEARS = 8
rng = np.random.default_rng(20261117)
s = pd.read_csv(C.INTERIM / "phenology_site_year.csv")

# 1. 수목원 x 종 x 단계 ------------------------------------------------
def mad_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))

g = s.groupby(["현상", "종구분", "종", "수목원"])["날짜숫자"]
t = g.agg(관측연수="count", 평균="mean", 표준편차="std", 로버스트표준편차=mad_sd,
          가장이른="min", 가장늦은="max").reset_index()
t["범위"] = t["가장늦은"] - t["가장이른"]
t["평균날짜"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(t["평균"].round() - 1, unit="D")
t["평균날짜"] = t["평균날짜"].dt.strftime("%m-%d")
t["연수충족"] = t["관측연수"] >= MIN_YEARS
t.round(2).to_csv(C.TABLES / "variability_site_species.csv", index=False, encoding="utf-8-sig")

# 2. 종별·수목원별 변동 폭 ---------------------------------------------
TARGETS = {"개화시작": C.SPRING_SPECIES, "개화50%이상": C.SPRING_SPECIES, "낙화90~100%": C.SPRING_SPECIES,
           "단풍50%이상": C.AUTUMN_SPECIES, "단풍90~100%": C.AUTUMN_SPECIES}


def pooled_sd(dev, k):
    """그룹 평균을 뺀 편차 dev, 그룹 수 k -> 합동 표준편차"""
    return np.sqrt((dev ** 2).sum() / max(len(dev) - k, 1))


def boot_ci(df, n=2000):
    """수목원·종 묶음 안에서 연도를 복원추출해 합동 표준편차의 95% 구간"""
    groups = [x["날짜숫자"].to_numpy() for _, x in df.groupby(["수목원", "종"])]
    out = []
    for _ in range(n):
        ss, dof = 0.0, 0
        for x in groups:
            b = rng.choice(x, len(x))
            # 복원추출 표본의 분산은 (n-1)/n만큼 작게 나오므로 n/(n-1)로 보정
            ss += ((b - b.mean()) ** 2).sum() * len(x) / (len(x) - 1)
            dof += len(x) - 1
        out.append(np.sqrt(ss / dof))
    return np.percentile(out, [2.5, 97.5])


def bf_stat(absdev, labels):
    """Brown-Forsythe: 중앙값에서 벗어난 크기를 그룹 간 분산분석한 F"""
    df = pd.DataFrame({"z": absdev, "g": labels})
    gm = df.groupby("g")["z"].agg(["mean", "count"])
    grand = df["z"].mean()
    ssb = (gm["count"] * (gm["mean"] - grand) ** 2).sum()
    ssw = ((df["z"] - df["g"].map(gm["mean"])) ** 2).sum()
    k = len(gm)
    return (ssb / (k - 1)) / (ssw / (len(df) - k))


def perm_test(d, factor, within, n=3000):
    """factor(종 또는 수목원) 간 변동 폭 차이. within 안에서만 라벨을 섞는다."""
    obs = bf_stat(d["절대편차"].to_numpy(), d[factor].to_numpy())
    cnt = 0
    idx = [np.flatnonzero((d[within] == w).to_numpy()) for w in d[within].unique()]
    lab = d[factor].to_numpy().copy()
    for _ in range(n):
        p = lab.copy()
        for i in idx:
            p[i] = rng.permutation(lab[i])
        cnt += bf_stat(d["절대편차"].to_numpy(), p) >= obs
    return obs, (cnt + 1) / (n + 1)


sp_rows, site_rows, tests = [], [], []
for stage, species in TARGETS.items():
    d = s[(s["현상"] == stage) & s["종"].isin(species)].copy()
    n = d.groupby(["수목원", "종"])["날짜숫자"].transform("count")
    d = d[n >= MIN_YEARS].copy()
    grp = d.groupby(["수목원", "종"])["날짜숫자"]
    d["편차"] = d["날짜숫자"] - grp.transform("mean")
    d["절대편차"] = (d["날짜숫자"] - grp.transform("median")).abs()
    for factor, rows in (("종", sp_rows), ("수목원", site_rows)):
        for name, x in d.groupby(factor):
            k = x.groupby(["수목원", "종"]).ngroups
            lo, hi = boot_ci(x)
            cell = x.groupby(["수목원", "종"])["날짜숫자"].std()
            rows.append({"현상": stage, factor: name, "묶음수": k, "관측수": len(x),
                         "평균날짜숫자": round(x["날짜숫자"].mean(), 1),
                         "변동폭_표준편차": round(pooled_sd(x["편차"], k), 2),
                         "95%하한": round(lo, 2), "95%상한": round(hi, 2),
                         "묶음별_최소": round(cell.min(), 2), "묶음별_최대": round(cell.max(), 2)})
    f1, p1 = perm_test(d, "종", "수목원")
    f2, p2 = perm_test(d, "수목원", "종")
    cell = d.groupby(["수목원", "종"])["날짜숫자"].std()
    tests.append({"현상": stage, "묶음수": len(cell), "관측수": len(d),
                  "묶음별표준편차_최소": round(cell.min(), 2), "묶음별표준편차_중앙": round(cell.median(), 2),
                  "묶음별표준편차_최대": round(cell.max(), 2),
                  "종간_F": round(f1, 2), "종간_p": round(p1, 4),
                  "수목원간_F": round(f2, 2), "수목원간_p": round(p2, 4)})

sp = pd.DataFrame(sp_rows); st = pd.DataFrame(site_rows); te = pd.DataFrame(tests)
sp.to_csv(C.TABLES / "variability_by_species.csv", index=False, encoding="utf-8-sig")
st.to_csv(C.TABLES / "variability_by_site.csv", index=False, encoding="utf-8-sig")
te.to_csv(C.TABLES / "variability_tests.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(te.to_string(index=False))
for stage in ("개화시작", "단풍50%이상", "단풍90~100%"):
    print("\n==", stage)
    print(sp[sp["현상"] == stage].sort_values("변동폭_표준편차").to_string(index=False))
    print(st[st["현상"] == stage].sort_values("변동폭_표준편차").to_string(index=False))
m = t[(t["현상"] == "개화시작") & t["종"].isin(C.SPRING_SPECIES)]
print(m.pivot(index="종", columns="수목원", values="표준편차").round(1).to_string())
print(m.pivot(index="종", columns="수목원", values="관측연수").to_string())
