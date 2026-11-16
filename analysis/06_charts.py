"""차트 10장을 reports/figures/에 PNG로 저장한다. 먼저 01~05 스크립트를 실행해 둔다."""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import config as C

# ---- 공통 모양 --------------------------------------------------------
for f in font_manager.findSystemFonts():
    low = f.lower()
    if "notosanscjk" in low or "nanumgothic" in low or "malgun" in low:
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
names = sorted({f.name for f in font_manager.fontManager.ttflist})
# Noto Sans CJK는 묶음 파일의 첫 서체 이름(JP 등)으로 잡히지만 한글 글자를 모두 담고 있다
FONT = next((n for n in names if n.startswith("Noto Sans CJK") or n in ("NanumGothic", "Malgun Gothic")), "sans-serif")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
plt.rcParams.update({
    "font.family": FONT, "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "xtick.major.size": 0, "ytick.major.size": 0})
SRC = "자료: 국립수목원 식물계절관측 DB(2009~2025) 전시원 10곳, 기상청 ASOS 일자료 · 8년 이상 관측된 수목원·종 조합"


def finish(fig, title, sub, name, src=SRC, top=0.86):
    fig.text(0.035, 0.955, title, fontsize=14.5, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.905, sub, fontsize=10, color=INK2, ha="left", va="top")
    fig.text(0.035, 0.018, src, fontsize=7.8, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(top=top, bottom=0.13)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / name, dpi=200)
    plt.close(fig)


def md(doy):
    d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=float(doy) - 1)
    return f"{d.month}/{d.day}"


def date_axis(ax, lo, hi, step=10, axis="x"):
    ticks = [pd.Timestamp(2025, m, d).dayofyear for m in range(1, 13) for d in (1, 11, 21)]
    ticks = [t for t in ticks if lo <= t <= hi]
    if step > 10:
        ticks = [t for t in ticks if pd.Timestamp("2025-01-01") + pd.Timedelta(days=t - 1) and
                 (pd.Timestamp("2025-01-01") + pd.Timedelta(days=t - 1)).day == 1]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([md(t) for t in ticks])


s = pd.read_csv(C.INTERIM / "phenology_site_year.csv")
w = pd.read_csv(C.INTERIM / "weather_daily.csv", parse_dates=["일시"])
mp = pd.read_csv(C.REFERENCE / "station_map.csv")
vsp = pd.read_csv(C.TABLES / "variability_by_species.csv")
vst = pd.read_csv(C.TABLES / "variability_by_site.csv")
tests = pd.read_csv(C.TABLES / "variability_tests.csv").set_index("현상")
SITE_SHORT = lambda x: x.replace("수목원", "").replace("강원도립화목원", "강원도립화목원")


def dot_ci(ax, df, label, color=BLUE):
    df = df.sort_values("변동폭_표준편차", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    ax.hlines(y, df["95%하한"], df["95%상한"], color=color, lw=2, alpha=0.45)
    ax.scatter(df["변동폭_표준편차"], y, s=62, color=color, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for yi, v, hi in zip(y, df["변동폭_표준편차"], df["95%상한"]):
        ax.text(hi + 0.25, yi, f"{v:.1f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(df[label]); ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(df) - 0.3)


# ---- 1. 종별 변동 폭 ---------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 5.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.42})
d1 = vsp[vsp["현상"] == "개화50%이상"]; d2 = vsp[vsp["현상"] == "단풍50%이상"]
dot_ci(a1, d1, "종"); dot_ci(a2, d2, "종", ORANGE)
for a, t in ((a1, "봄꽃 11종 · 개화50%이상"), (a2, "단풍 5종 · 단풍50%이상")):
    a.set_xlim(0, 16.5); a.set_xlabel("해마다 날짜의 표준편차(일)"); a.set_title(t, fontsize=10.5, loc="left", color=INK2, pad=8)
finish(fig, "종에 따라 해마다 흔들리는 폭이 다르다",
       f"봄꽃은 4.9~7.7일로 약 1.6배 차이, 단풍은 은행나무만 뚜렷하게 크다 · 선은 95% 구간 (종 간 차이 p={tests.loc['개화50%이상','종간_p']:.4f}, {tests.loc['단풍50%이상','종간_p']:.2f})",
       "fig01_species_variability.png", top=0.80)

# ---- 2. 개화 시기와 변동 폭 --------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.6))
x, y = d1["평균날짜숫자"].to_numpy(), d1["변동폭_표준편차"].to_numpy()
k, c0 = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
xx = np.array([x.min() - 3, x.max() + 3]); ax.plot(xx, c0 + k * xx, color=GRAY, lw=2, zorder=1)
ax.scatter(x, y, s=80, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
OFF = {"백목련": (6, 2), "개나리": (-6, -15), "진달래": (6, 5), "산수유": (-8, 5), "생강나무": (-8, 6), "미선나무": (6, -12),
       "왕벚나무": (7, -3), "철쭉": (7, 4), "산철쭉": (-8, -12), "마가목": (7, 5), "아까시나무": (-8, 8)}
for _, rw in d1.iterrows():
    dx, dy = OFF.get(rw["종"], (6, 4))
    ax.annotate(rw["종"], (rw["평균날짜숫자"], rw["변동폭_표준편차"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=9.5, ha="left" if dx > 0 else "right")
date_axis(ax, 75, 142); ax.set_xlim(76, 142); ax.set_ylim(4, 8.4)
ax.set_xlabel("평균 개화일 (개화50%이상, 수목원 평균)"); ax.set_ylabel("해마다 날짜의 표준편차(일)")
finish(fig, "늦게 피는 종일수록 개화 시기가 안정적이다",
       f"3월에 피는 종은 6~8일, 5월에 피는 종은 5일 안팎으로 흔들린다 · 상관계수 {r:.2f} (봄꽃 11종)",
       "fig02_timing_vs_variability.png")

# ---- 3. 수목원별 변동 폭 -----------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.6))
d3 = vst[vst["현상"] == "개화50%이상"].copy()
dot_ci(ax, d3, "수목원")
lo, hi = d3["95%하한"].max(), d3["95%상한"].min()
ax.set_xlim(0, 16.5); ax.set_xlabel("해마다 날짜의 표준편차(일) · 개화50%이상, 봄꽃 11종을 합쳐 계산")
finish(fig, "지역에 따른 변동 폭 차이는 확인되지 않는다",
       f"수목원별 5.6~7.7일이지만 95% 구간이 서로 겹친다 (지역 간 차이 p={tests.loc['개화50%이상','수목원간_p']:.2f}) · 가로축은 종별 차트와 같은 눈금",
       "fig03_site_variability.png")

# ---- 4. 연도 효과 ------------------------------------------------------
f = s[(s["현상"] == "개화50%이상") & s["종"].isin(C.SPRING_SPECIES)].copy()
f = f[f.groupby(["수목원", "종"])["날짜숫자"].transform("count") >= C.MIN_YEARS]
f["편차"] = f["날짜숫자"] - f.groupby(["수목원", "종"])["날짜숫자"].transform("mean")
sy = f.groupby(["수목원", "연도"])["편차"].agg(["mean", "count"]).reset_index()
sy = sy[sy["count"] >= 3]
nat = f.groupby("연도")["편차"].mean()
tot = (f["편차"] ** 2).sum()
share = 1 - ((f["편차"] - f.groupby("연도")["편차"].transform("mean")) ** 2).sum() / tot
fig, ax = plt.subplots(figsize=(9.6, 5.6))
ax.axhline(0, color=INK2, lw=0.9)
for i, (site, g) in enumerate(sy.groupby("수목원")):
    ax.plot(g["연도"], g["mean"], color=GRAY, lw=1.3, alpha=0.9, label="수목원 10곳 각각" if i == 0 else None)
nat2 = nat[nat.index >= 2010]
ax.plot(nat2.index, nat2.values, color=BLUE, lw=2.6, marker="o", ms=5.5, mec=SURFACE, mew=1.2, label="전체 평균", zorder=4)
for yv in (2011, 2021):
    ax.annotate(f"{yv}년 {nat[yv]:+.1f}일", (yv, nat[yv]), xytext=(0, 12 if nat[yv] > 0 else -17), textcoords="offset points",
                ha="center", fontsize=9.5, color=INK)
ax.set_xticks(range(2009, 2026, 2)); ax.set_xlim(2008.6, 2025.4)
ax.set_ylabel("평년보다 늦음(+) / 빠름(-), 일"); ax.legend(frameon=False, loc="upper right", fontsize=9.5)
finish(fig, "어느 해에 빠르고 늦은지는 전국이 함께 움직인다",
       f"개화일 흔들림(분산)의 {share:.0%}는 모든 수목원에 공통인 연도 효과다 · 각 수목원·종의 17년 평균을 0으로 둔 편차",
       "fig04_year_effect.png")

# ---- 5. 기온과 개화일 --------------------------------------------------
t23 = w[w["일시"].dt.month.isin([2, 3])].groupby(["지점명", "연도"])["평균기온"].mean().rename("기온").reset_index()
z = sy.merge(mp[["수목원", "기상지점명"]]).merge(t23, left_on=["기상지점명", "연도"], right_on=["지점명", "연도"])
z["기온편차"] = z["기온"] - z.groupby("수목원")["기온"].transform("mean")
k, c0 = np.polyfit(z["기온편차"], z["mean"], 1); r = np.corrcoef(z["기온편차"], z["mean"])[0, 1]
fig, ax = plt.subplots(figsize=(9.6, 5.6))
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.scatter(z["기온편차"], z["mean"], s=44, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=1, zorder=3)
xx = np.array([z["기온편차"].min(), z["기온편차"].max()]); ax.plot(xx, c0 + k * xx, color=INK, lw=1.8, zorder=4)
ax.set_xlabel("2~3월 평균기온, 그 수목원의 평년 대비(도)"); ax.set_ylabel("개화일, 평년보다 늦음(+) / 빠름(-), 일")
finish(fig, f"2~3월이 1도 따뜻하면 개화가 약 {abs(k):.1f}일 빨라진다",
       f"점 하나는 수목원 한 곳의 한 해 ({len(z)}개) · 상관계수 {r:.2f} · 기온은 연결한 기상 지점 값",
       "fig05_temp_vs_bloom.png")
pd.DataFrame([{"기울기_일per도": round(k, 2), "상관계수": round(r, 2), "점수": len(z), "공통연도효과_분산비율": round(share, 3)}]).to_csv(
    C.TABLES / "eda_temp_pooled.csv", index=False, encoding="utf-8-sig")

# ---- 6. 개화 달력 ------------------------------------------------------
P = s.pivot(index=["수목원", "종", "연도"], columns="현상", values="날짜숫자").reset_index()
b = P[P["종"].isin(C.SPRING_SPECIES)].dropna(subset=["개화시작", "낙화90~100%"]).copy()
b = b[b["낙화90~100%"] > b["개화시작"]]
cal = b.groupby("종").agg(시작=("개화시작", "mean"), 끝=("낙화90~100%", "mean"), 이른=("개화시작", lambda x: x.quantile(0.1)),
                         늦은=("낙화90~100%", lambda x: x.quantile(0.9))).sort_values("시작", ascending=False)
fig, ax = plt.subplots(figsize=(9.6, 5.6))
y = np.arange(len(cal))
ax.barh(y, cal["늦은"] - cal["이른"], left=cal["이른"], height=0.5, color="#cde2fb", label="이른 해의 시작 ~ 늦은 해의 끝 (10~90%)")
ax.barh(y, cal["끝"] - cal["시작"], left=cal["시작"], height=0.5, color=BLUE, label="평균 개화시작 ~ 평균 낙화")
ax.set_yticks(y); ax.set_yticklabels(cal.index); ax.grid(axis="y", visible=False)
date_axis(ax, 40, 170); ax.set_xlim(42, 168); ax.set_ylim(-0.7, len(cal) - 0.3)
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
finish(fig, "봄꽃 11종의 개화 달력", "전시원 10곳 전체 평균 · 진한 막대가 평균적인 개화 기간, 옅은 막대가 해와 지역에 따라 꽃이 있을 수 있는 범위",
       "fig06_bloom_calendar.png", src="자료: 국립수목원 식물계절관측 DB(2009~2025) 전시원 10곳 · 개화시작과 낙화90~100%가 모두 기록된 수목원·종·해")

# ---- 7. 개화 기간 ------------------------------------------------------
dur = pd.read_csv(C.TABLES / "eda_bloom_duration_species.csv").sort_values("중앙값", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.6, 5.6))
y = np.arange(len(dur))
ax.hlines(y, dur["하위10"], dur["상위10"], color=BLUE, lw=2, alpha=0.45)
ax.scatter(dur["중앙값"], y, s=70, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
for yi, v, hi in zip(y, dur["중앙값"], dur["상위10"]):
    ax.text(hi + 0.7, yi, f"{v:.0f}일", va="center", fontsize=9, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(dur["종"]); ax.grid(axis="y", visible=False)
ax.set_xlim(0, 40); ax.set_ylim(-0.7, len(dur) - 0.3); ax.set_xlabel("개화시작부터 낙화90~100%까지(일)")
_m = cal.join(dur.set_index("종")["중앙값"])
r_dur = np.corrcoef(_m["시작"], _m["중앙값"])[0, 1]
finish(fig, "일찍 피는 종이 대체로 더 오래 핀다", f"생강나무·산수유·개나리는 3~4주, 왕벚나무·백목련·마가목은 2주 · 평균 개화시작일과의 상관계수 {r_dur:.2f} · 점은 중앙값, 선은 10~90% 범위",
       "fig07_bloom_duration.png", src="자료: 국립수목원 식물계절관측 DB(2009~2025) 전시원 10곳 · 개화시작과 낙화90~100%가 모두 기록된 수목원·종·해")

# ---- 8. 날짜를 고정했을 때 적중률 ---------------------------------------
hs = pd.read_csv(C.TABLES / "eda_fixed_day_hit_species.csv").sort_values("하루_적중률").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.6, 5.6))
y = np.arange(len(hs))
ax.hlines(y, hs["하루_최소"] * 100, hs["하루_최대"] * 100, color=BLUE, lw=2, alpha=0.4)
ax.scatter(hs["하루_적중률"] * 100, y, s=70, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
for yi, v, sd, du in zip(y, hs["하루_적중률"], hs["개화시작_표준편차"], hs["개화기간_중앙값"]):
    ax.text(102, yi, f"{v*100:.0f}%   변동 {sd:.1f}일 · 기간 {du:.0f}일", va="center", fontsize=9, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(hs["종"]); ax.grid(axis="y", visible=False)
ax.set_xlim(50, 128); ax.set_xticks([50, 60, 70, 80, 90, 100]); ax.set_xticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
ax.set_ylim(-0.7, len(hs) - 0.3); ax.set_xlabel("가장 유리한 하루를 골랐을 때 그날 꽃이 피어 있던 해의 비율")
finish(fig, "축제 날짜를 고정하면, 오래 피는 종이 더 잘 맞는다",
       "변동 폭이 더 큰 산수유·개나리가 변동 폭이 작은 왕벚나무보다 적중률이 높다 · 점은 수목원 평균, 선은 수목원별 최소~최대",
       "fig08_fixed_day_hit.png", src="자료: 국립수목원 식물계절관측 DB(2009~2025) 전시원 10곳 · 개화시작~낙화가 8년 이상 기록된 조합 · 날짜는 사후에 고른 최적일이라 실제 적중률은 이보다 낮다")

# ---- 9. 종 사이 간격 ---------------------------------------------------
gy = pd.read_csv(C.TABLES / "eda_gap_by_year.csv", index_col=0)
gp = pd.read_csv(C.TABLES / "eda_species_gaps.csv")
fig, ax = plt.subplots(figsize=(9.6, 5.6))
SER = [("산수유→왕벚나무", BLUE, "o"), ("왕벚나무→산철쭉", ORANGE, "s"), ("왕벚나무→아까시나무", AQUA, "^")]
for name, col, mk in SER:
    v = gy[name].dropna()
    ax.plot(v.index, v.values, color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.2, label=name)
    a, c = name.split("→"); row = gp[(gp["먼저피는종"] == a) & (gp["나중피는종"] == c)].iloc[0]
    ax.text(2025.35, v.iloc[-1], f"{name}\n평균 {row['평균간격']:.0f}일, 표준편차 {row['간격_표준편차']:.1f}일", va="center", fontsize=9, color=INK)
ax.set_xticks(range(2010, 2026, 3)); ax.set_xlim(2009.5, 2030.5); ax.set_ylim(0, 50)
ax.set_ylabel("두 종의 개화50%이상 날짜 차이(일)"); ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9.5, bbox_to_anchor=(0, 1.07))
finish(fig, "피는 순서는 유지되지만 간격은 해마다 4~5일씩 흔들린다",
       "왕벚나무와의 간격 · 같은 수목원에서 두 종이 모두 기록된 해만 사용, 수목원 3곳 이상의 평균",
       "fig09_species_gaps.png", top=0.82)

# ---- 10. 봄과 가을의 추세 ----------------------------------------------
an = pd.read_csv(C.TABLES / "eda_year_anomaly.csv", index_col=0)
tr = pd.read_csv(C.TABLES / "eda_trends.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.6), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, stage, col, ttl in ((axes[0], "개화50%이상", BLUE, "봄: 개화50%이상 (봄꽃 11종)"), (axes[1], "단풍50%이상", ORANGE, "가을: 단풍50%이상 (16종)")):
    v = an[stage]; v = v[v.index >= 2010]
    ax.axhline(0, color=INK2, lw=0.9)
    ax.bar(v.index, v.values, width=0.62, color=col)
    X = v.index.to_numpy(float); k, c0 = np.polyfit(X, v.values, 1)
    ax.plot(X, c0 + k * X, color=INK, lw=1.6, ls=(0, (4, 3)))
    t = tr[(tr["현상"] == stage) & (tr["기간"] == "2010~2025")].iloc[0]; t2 = tr[(tr["현상"] == stage) & (tr["기간"] == "2014~2025")].iloc[0]
    ax.set_title(ttl, fontsize=10.5, loc="left", color=INK2, pad=8)
    ax.text(0.03, 0.04 if stage == "개화50%이상" else 0.93, f"10년에 {t['10년당_일']:+.1f}일 (±{t['95%오차']:.1f})\n2014년 이후만 보면 {t2['10년당_일']:+.1f}일 (±{t2['95%오차']:.1f})",
            transform=ax.transAxes, fontsize=9.5, va="bottom" if stage == "개화50%이상" else "top")
    ax.set_xticks(range(2010, 2026, 3)); ax.grid(axis="x", visible=False); ax.set_ylim(-10, 12.5)
axes[0].set_ylabel("평년보다 늦음(+) / 빠름(-), 일")
finish(fig, "봄은 빨라지고 단풍은 늦어지는 방향 (기존 연구와 같은 결과)",
       "전체 평균 연도 편차와 추세선 · 봄의 추세는 2010~2013년의 늦은 봄에 크게 좌우되고, 가을의 추세는 기간을 바꿔도 유지된다",
       "fig10_trend_spring_autumn.png", top=0.80)
print("saved", sorted(p.name for p in C.FIGURES.glob("*.png")), "| font", FONT)
print("duration-onset r", round(r_dur, 2))
