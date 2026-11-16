"""기상 자료를 정리하고, 전시원 10곳과 기상 지점 연결표를 만든다.

출력
- data/interim/weather_daily.csv   : 30개 지점 일자료 (저장소에 올리지 않음)
- reference/station_map.csv        : 전시원 10곳 x 기상 지점 연결표 (10줄)
- reports/tables/station_distance.csv : 전시원별 가까운 지점 3곳과 거리
- reports/tables/weather_missing.csv  : 지점별 결측 일수
지점 좌표는 reference/asos_stations.csv (기상청 관측 지점 정보)
"""
import numpy as np
import pandas as pd
import config as C

# 1. 기상 자료 ---------------------------------------------------------
w = pd.concat([pd.read_csv(f, encoding="cp949") for f in sorted(C.RAW_WEATHER.glob("*.csv"))],
              ignore_index=True)
w.columns = ["지점", "지점명", "일시", "평균기온", "최저기온", "최고기온",
             "일강수량", "일조시간", "평균지면온도"]
w["일시"] = pd.to_datetime(w["일시"])
assert not w.duplicated(["지점", "일시"]).any()
w["일강수량"] = w["일강수량"].fillna(0.0)      # 빈칸은 비가 오지 않은 날

# 지점별로 빠진 날짜를 줄로 만들어 넣는다 (값은 빈칸)
full = []
for (sid, name), g in w.groupby(["지점", "지점명"]):
    idx = pd.date_range(g["일시"].min(), g["일시"].max(), freq="D")
    g = g.set_index("일시").reindex(idx)
    g["지점"], g["지점명"] = sid, name
    g["날짜없음"] = g["평균기온"].isna() & g["최고기온"].isna() & g["일조시간"].isna() & g["일강수량"].isna()
    full.append(g.rename_axis("일시").reset_index())
w = pd.concat(full, ignore_index=True)
w["연도"] = w["일시"].dt.year
w["날짜숫자"] = w["일시"].dt.dayofyear
vals = ["평균기온", "최저기온", "최고기온", "일강수량", "일조시간", "평균지면온도"]
miss = w.groupby(["지점", "지점명"]).agg(
    시작=("일시", "min"), 종료=("일시", "max"), 일수=("일시", "size"),
    날짜없음=("날짜없음", "sum"), **{f"{v}_결측": (v, lambda s: int(s.isna().sum())) for v in vals}).reset_index()
w = w[["지점", "지점명", "일시", "연도", "날짜숫자"] + vals]
w["일시"] = w["일시"].dt.strftime("%Y-%m-%d")
w.to_csv(C.INTERIM / "weather_daily.csv", index=False, encoding="utf-8-sig")
miss.to_csv(C.TABLES / "weather_missing.csv", index=False, encoding="utf-8-sig")
print(miss.to_string(index=False))

# 2. 지점 연결표 -------------------------------------------------------
st = pd.read_csv(C.REFERENCE / "asos_stations.csv")
st["자료시작"] = st["지점"].map(miss.set_index("지점")["시작"])
FULL_START = pd.Timestamp("2008-12-01")

# 전시원 좌표: 세부관측이 전시원인 줄의 위도·경도 중앙값 (먼 좌표 몇 줄에 영향받지 않음)
frames = [pd.read_excel(f, dtype=str) for f in sorted(C.RAW_PHENO.glob("*.xlsx"))]
p = pd.concat(frames, ignore_index=True)
p = p[p["세부관측"] == C.SITE_SCOPE].copy()
p["위도"], p["경도"], p["연도"] = p["위도"].astype(float), p["경도"].astype(float), p["연도"].astype(int)
site = p.groupby("관측자").agg(위도=("위도", "median"), 경도=("경도", "median"),
                              관측시작연도=("연도", "min"), 관측종료연도=("연도", "max")).reset_index()


def km(lat1, lon1, lat2, lon2):
    r = np.radians
    a = np.sin((r(lat2) - r(lat1)) / 2) ** 2 + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin((r(lon2) - r(lon1)) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


rows, near = [], []
for _, s in site.iterrows():
    d = st.assign(거리_km=km(s["위도"], s["경도"], st["위도"], st["경도"])).sort_values("거리_km")
    for rank, (_, x) in enumerate(d.head(3).iterrows(), 1):
        near.append({"수목원": s["관측자"], "순위": rank, "지점": x["지점"], "지점명": x["지점명"],
                     "거리_km": round(x["거리_km"], 1), "해발고도_m": x["해발고도_m"],
                     "자료시작": x["자료시작"].date()})
    best_all = d.iloc[0]
    # 주 지점: 2008년 12월부터 자료가 있는 지점 중 가장 가까운 곳
    best = d[d["자료시작"] <= FULL_START].iloc[0]
    note = ""
    if best_all["지점"] != best["지점"]:
        note = (f"가장 가까운 지점은 {best_all['지점명']}({best_all['거리_km']:.1f}km)이나 "
                f"{best_all['자료시작'].year}년부터만 자료가 있어 제외")
    rows.append({"수목원": s["관측자"], "수목원_위도": round(s["위도"], 5), "수목원_경도": round(s["경도"], 5),
                 "관측시작연도": s["관측시작연도"], "관측종료연도": s["관측종료연도"],
                 "기상지점": best["지점"], "기상지점명": best["지점명"],
                 "기상지점_위도": best["위도"], "기상지점_경도": best["경도"],
                 "기상지점_해발고도_m": best["해발고도_m"], "거리_km": round(best["거리_km"], 1), "비고": note})
m = pd.DataFrame(rows)
m.to_csv(C.REFERENCE / "station_map.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(near).to_csv(C.TABLES / "station_distance.csv", index=False, encoding="utf-8-sig")
print(m.to_string(index=False))
print(pd.DataFrame(near).to_string(index=False))
