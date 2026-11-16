"""탐색 분석: 개화 기간, 고정 날짜 적중률, 종 사이 간격, 추세.

출력 (reports/tables/)
- eda_bloom_duration_species.csv / eda_bloom_duration_site.csv : 개화시작~낙화 일수
- eda_fixed_day_hit.csv   : 날짜를 고정했을 때 꽃이 피어 있던 해의 비율 (수목원 x 종, 종별 요약)
- eda_species_gaps.csv    : 종 사이 개화 간격과 그 변동
- eda_gap_by_year.csv     : 연도별 간격 (전국 평균)
- eda_year_anomaly.csv    : 단계별 연도 편차 (전국 평균)
- eda_trends.csv          : 단계별 추세 (10년당 일수)
"""
import math
import numpy as np
import pandas as pd
import config as C

s = pd.read_csv(C.INTERIM / "phenology_site_year.csv")
w = pd.read_csv(C.INTERIM / "weather_daily.csv", parse_dates=["일시"])
mp = pd.read_csv(C.REFERENCE / "station_map.csv")
P = s.pivot(index=["수목원", "종", "연도"], columns="현상", values="날짜숫자").reset_index()

# 1. 개화 기간 ---------------------------------------------------------
b = P[P["종"].isin(C.SPRING_SPECIES)].dropna(subset=["개화시작", "낙화90~100%"]).copy()
b["개화기간"] = b["낙화90~100%"] - b["개화시작"]
b = b[b["개화기간"] > 0]


def q(p):
    return lambda x: x.quantile(p)


def dur_table(key):
    out = b.groupby(key)["개화기간"].agg(관측수="count", 중앙값="median", 평균="mean", 표준편차="std",
                                      하위10="quantile", 상위10="quantile").reset_index()
    out["하위10"] = b.groupby(key)["개화기간"].quantile(0.1).values
    out["상위10"] = b.groupby(key)["개화기간"].quantile(0.9).values
    return out.round(1).sort_values("중앙값")


dur_sp = dur_table("종"); dur_site = dur_table("수목원")
dur_sp.to_csv(C.TABLES / "eda_bloom_duration_species.csv", index=False, encoding="utf-8-sig")
dur_site.to_csv(C.TABLES / "eda_bloom_duration_site.csv", index=False, encoding="utf-8-sig")
b["기간편차"] = b["개화기간"] - b.groupby(["수목원", "종"])["개화기간"].transform("mean")
b["시작편차"] = b["개화시작"] - b.groupby(["수목원", "종"])["개화시작"].transform("mean")
r_dur_onset = np.corrcoef(b["기간편차"], b["시작편차"])[0, 1]
slope_dur_onset = np.polyfit(b["시작편차"], b["기간편차"], 1)[0]

# 2. 고정 날짜 적중률 ---------------------------------------------------
rows = []
for (site, sp), g in b.groupby(["수목원", "종"]):
    if len(g) < C.MIN_YEARS:
        continue
    days = np.arange(1, 201)
    on, off = g["개화시작"].to_numpy()[:, None], g["낙화90~100%"].to_numpy()[:, None]
    inb = (days >= on) & (days <= off)                     # 연도 x 날짜
    one = inb.mean(axis=0)
    three = (inb[:, :-2] & inb[:, 1:-1] & inb[:, 2:]).mean(axis=0)
    rows.append({"수목원": site, "종": sp, "관측연수": len(g),
                 "개화시작_표준편차": round(g["개화시작"].std(), 2), "개화기간_중앙값": g["개화기간"].median(),
                 "최적고정일": int(days[one.argmax()]), "하루_적중률": round(one.max(), 3),
                 "사흘연속_적중률": round(three.max(), 3)})
hit = pd.DataFrame(rows)
hit_sp = hit.groupby("종").agg(수목원수=("수목원", "size"), 개화시작_표준편차=("개화시작_표준편차", "mean"),
                              개화기간_중앙값=("개화기간_중앙값", "mean"), 하루_적중률=("하루_적중률", "mean"),
                              사흘연속_적중률=("사흘연속_적중률", "mean"), 하루_최소=("하루_적중률", "min"),
                              하루_최대=("하루_적중률", "max")).round(3).reset_index().sort_values("하루_적중률")
hit.to_csv(C.TABLES / "eda_fixed_day_hit.csv", index=False, encoding="utf-8-sig")
hit_sp.to_csv(C.TABLES / "eda_fixed_day_hit_species.csv", index=False, encoding="utf-8-sig")

# 3. 종 사이 간격 (개화50%이상) ------------------------------------------
F = s[(s["현상"] == "개화50%이상") & s["종"].isin(C.SPRING_SPECIES)].pivot(
    index=["수목원", "연도"], columns="종", values="날짜숫자")
PAIRS = [("산수유", "왕벚나무"), ("개나리", "왕벚나무"), ("진달래", "왕벚나무"), ("백목련", "왕벚나무"),
         ("왕벚나무", "철쭉"), ("왕벚나무", "산철쭉"), ("왕벚나무", "아까시나무"), ("산수유", "아까시나무")]
rows, by_year = [], {}
sd_single = {}
for sp in C.SPRING_SPECIES:
    x = F[sp].dropna().reset_index()
    x["편차"] = x[sp] - x.groupby("수목원")[sp].transform("mean")
    sd_single[sp] = math.sqrt((x["편차"] ** 2).sum() / (len(x) - x["수목원"].nunique()))
for a, c in PAIRS:
    g = (F[c] - F[a]).dropna().rename("간격").reset_index()
    g["편차"] = g["간격"] - g.groupby("수목원")["간격"].transform("mean")
    rows.append({"먼저피는종": a, "나중피는종": c, "관측수": len(g), "평균간격": round(g["간격"].mean(), 1),
                 "간격_표준편차": round(math.sqrt((g["편차"] ** 2).sum() / (len(g) - g["수목원"].nunique())), 2),
                 "나중종_날짜_표준편차": round(sd_single[c], 2),
                 "최소": g["간격"].min(), "최대": g["간격"].max(), "순서뒤집힘": int((g["간격"] < 0).sum())})
    yy = g.groupby("연도")["간격"].agg(["mean", "count"])
    by_year[f"{a}→{c}"] = yy.loc[yy["count"] >= 3, "mean"]
gaps = pd.DataFrame(rows)
gaps.to_csv(C.TABLES / "eda_species_gaps.csv", index=False, encoding="utf-8-sig")
gy = pd.DataFrame(by_year).round(1)
gy.to_csv(C.TABLES / "eda_gap_by_year.csv", encoding="utf-8-sig")

# 4. 연도 편차와 추세 ---------------------------------------------------
ALL16 = C.SPRING_SPECIES + C.AUTUMN_SPECIES
STAGES = {"개화50%이상": C.SPRING_SPECIES, "개엽시작": ALL16, "단풍50%이상": ALL16,
          "단풍90~100%": ALL16, "낙엽90~100%": ALL16}
anom, tr = {}, []
for stage, species in STAGES.items():
    f = s[(s["현상"] == stage) & s["종"].isin(species)].copy()
    f = f[f.groupby(["수목원", "종"])["날짜숫자"].transform("count") >= C.MIN_YEARS]
    f["편차"] = f["날짜숫자"] - f.groupby(["수목원", "종"])["날짜숫자"].transform("mean")
    y = f.groupby("연도")["편차"].mean()
    anom[stage] = y
    for start in (2010, 2014):
        z = y[y.index >= start]
        X = z.index.to_numpy(float)
        k, c0 = np.polyfit(X, z.to_numpy(), 1)
        res = z.to_numpy() - (c0 + k * X)
        se = math.sqrt((res ** 2).sum() / (len(X) - 2) / ((X - X.mean()) ** 2).sum())
        tr.append({"현상": stage, "종수": len(species), "관측수": len(f), "기간": f"{start}~2025",
                   "10년당_일": round(k * 10, 1), "95%오차": round(1.96 * se * 10, 1)})
an = pd.DataFrame(anom).round(2)
an.to_csv(C.TABLES / "eda_year_anomaly.csv", encoding="utf-8-sig")
trd = pd.DataFrame(tr)
trd.to_csv(C.TABLES / "eda_trends.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(dur_sp.to_string(index=False)); print(dur_site.to_string(index=False))
print(f"개화기간과 개화시작 편차: r={r_dur_onset:.2f}, 시작이 하루 늦으면 기간 {slope_dur_onset:.2f}일")
print(hit_sp.to_string(index=False)); print(gaps.to_string(index=False)); print(gy.T.to_string()); print(an.T.to_string()); print(trd.to_string(index=False))
