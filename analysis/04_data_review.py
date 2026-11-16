"""자료 검토: 정제 기준을 정할 때 확인한 내용을 표로 남긴다.

출력 (reports/tables/)
- review_first_record_doy.csv : 수목원 x 연도별 그해 첫 기록의 날짜숫자 (관측을 늦게 시작한 해 확인)
- review_trees_per_year.csv   : 수목원 x 연도별 관측 나무 수 (2019년 증가 확인)
- review_tree_spread.csv      : 같은 수목원·종·해에서 나무 사이 날짜 차이
- review_temp_sensitivity.csv : 2~3월 평균기온과 개화일의 관계 (연결한 지점과 대안 지점 비교)
- review_station_offset.csv   : 연결 지점의 연평균기온이 지점 평균에서 벗어난 정도 (지점 이전 등 확인)
- review_year_effect.csv      : 연도별 개화일 편차 (수목원 x 연도)와 분산 분해
"""
import numpy as np
import pandas as pd
import config as C

t = pd.read_csv(C.INTERIM / "phenology_tree_year.csv", dtype={"나무번호": str})
s = pd.read_csv(C.INTERIM / "phenology_site_year.csv")
w = pd.read_csv(C.INTERIM / "weather_daily.csv", parse_dates=["일시"])
mp = pd.read_csv(C.REFERENCE / "station_map.csv")

# 1. 첫 기록 날짜, 나무 수 ---------------------------------------------
first = t.groupby(["수목원", "연도"])["날짜숫자"].min().unstack("연도")
first.to_csv(C.TABLES / "review_first_record_doy.csv", encoding="utf-8-sig")
trees = t.drop_duplicates(["수목원", "종", "나무번호", "연도"]).groupby(["수목원", "연도"]).size().unstack("연도")
trees.to_csv(C.TABLES / "review_trees_per_year.csv", encoding="utf-8-sig")

# 2. 나무 사이 날짜 차이 -----------------------------------------------
g = t.groupby(["수목원", "종", "연도", "현상"])["날짜숫자"].agg(["min", "max", "count"])
g = g[g["count"] > 1]
g["차이"] = g["max"] - g["min"]
sp = g.groupby("현상")["차이"].describe(percentiles=[0.5, 0.9]).round(1)
sp.to_csv(C.TABLES / "review_tree_spread.csv", encoding="utf-8-sig")

# 3. 기온과 개화일 -----------------------------------------------------
f = s[(s["현상"] == "개화50%이상") & (s["종구분"] == "봄꽃")].copy()
grp = f.groupby(["수목원", "종"])["날짜숫자"]
f = f[grp.transform("count") >= C.MIN_YEARS].copy()
f["편차"] = f["날짜숫자"] - f.groupby(["수목원", "종"])["날짜숫자"].transform("mean")
sy = f.groupby(["수목원", "연도"])["편차"].agg(평균편차="mean", 종수="count").reset_index()
t23 = w[w["일시"].dt.month.isin([2, 3])].groupby(["지점명", "연도"])["평균기온"].mean().rename("기온").reset_index()
ALT = {"미동산수목원": ["보은", "청주"], "대아수목원": ["금산", "전주"], "금강수목원": ["대전", "세종", "청주"],
       "경남수목원": ["진주", "의령군"], "국립수목원": ["동두천", "서울"], "강원도립화목원": ["춘천", "북춘천"],
       "물향기수목원": ["수원", "서울"], "대구수목원": ["대구"], "완도수목원": ["완도", "해남"],
       "한라수목원": ["제주", "서귀포"]}
rows = []
for site, sts in ALT.items():
    x = sy[(sy["수목원"] == site) & (sy["종수"] >= 3)]
    main = mp.loc[mp["수목원"] == site, "기상지점명"].iloc[0]
    for st in sts:
        z = x.merge(t23[t23["지점명"] == st], on="연도")
        if len(z) >= 5:
            rows.append({"수목원": site, "기상지점": st, "연결지점": st == main, "연수": len(z),
                         "상관계수": round(np.corrcoef(z["평균편차"], z["기온"])[0, 1], 2),
                         "기온1도당_일": round(np.polyfit(z["기온"], z["평균편차"], 1)[0], 1)})
sens = pd.DataFrame(rows)
sens.to_csv(C.TABLES / "review_temp_sensitivity.csv", index=False, encoding="utf-8-sig")

# 4. 지점별 연평균기온의 상대 위치 --------------------------------------
ids = mp["기상지점명"].tolist()
y = w[w["지점명"].isin(ids) & w["연도"].between(2009, 2025)].groupby(["연도", "지점명"])["평균기온"].mean().unstack()
off = y.sub(y.mean(axis=1), axis=0).round(2)
off.to_csv(C.TABLES / "review_station_offset.csv", encoding="utf-8-sig")

# 5. 연도 효과 ---------------------------------------------------------
ye = sy.pivot(index="수목원", columns="연도", values="평균편차").round(1)
ye.loc["전체"] = f.groupby("연도")["편차"].mean().round(1)
ye.to_csv(C.TABLES / "review_year_effect.csv", encoding="utf-8-sig")
tot = (f["편차"] ** 2).sum()
r_nat = 1 - ((f["편차"] - f.groupby("연도")["편차"].transform("mean")) ** 2).sum() / tot
r_site = 1 - ((f["편차"] - f.groupby(["수목원", "연도"])["편차"].transform("mean")) ** 2).sum() / tot

pd.set_option("display.width", 250)
print(first.to_string()); print(trees.to_string()); print(sp.loc[["개화시작", "개화50%이상", "단풍50%이상"]].to_string())
print(sens.to_string(index=False)); print(off.to_string()); print(ye.to_string())
print(f"개화50% 편차의 분산 중 전국 공통 연도 효과 {r_nat:.0%}, 수목원별 연도 효과까지 {r_site:.0%} (관측 {len(f)}건)")
print("전체 표준편차", round(f["편차"].std(), 2), "/ 전국 연도 효과 제거 후", round((f["편차"] - f.groupby("연도")["편차"].transform("mean")).std(), 2))
