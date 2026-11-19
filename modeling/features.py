# 기상 일자료를 수목원 x 연도 변수로 바꾼다
# 대구 지점 보정과 누출 점검(999)을 함께 둔다
# 누출 점검은 기준일 뒤 자료가 변수에 새어 들지 않는지 보는 것이다

import numpy as np
import pandas as pd

import mconfig as C

TEMP, RAIN, SUN = "평균기온", "일강수량", "일조시간"


def feb_end(y):
    return pd.Timestamp(y, 3, 1) - pd.Timedelta(days=1)


# 변수 이름마다 시작일과 끝일과 계산 방법과 원자료 열을 돌려준다
# 기간 창은 월과 일로 자른다(윤년은 2월 29일까지)
def windows(y):
    T, fe = pd.Timestamp, feb_end(y)
    return {
        "기온_12월": (T(y - 1, 12, 1), T(y - 1, 12, 31), "mean", TEMP),
        "기온_1월": (T(y, 1, 1), T(y, 1, 31), "mean", TEMP),
        "기온_2월": (T(y, 2, 1), fe, "mean", TEMP),
        "기온_3월": (T(y, 3, 1), T(y, 3, 31), "mean", TEMP),
        "기온_1_2월": (T(y, 1, 1), fe, "mean", TEMP),
        "기온_겨울_2월말": (T(y - 1, 12, 1), fe, "mean", TEMP),
        "기온_1_3월": (T(y, 1, 1), T(y, 3, 31), "mean", TEMP),
        "기온_2_3월": (T(y, 2, 1), T(y, 3, 31), "mean", TEMP),
        "강수_1_2월": (T(y, 1, 1), fe, "sum", RAIN),
        "강수_1_3월": (T(y, 1, 1), T(y, 3, 31), "sum", RAIN),
        "일조_1_2월": (T(y, 1, 1), fe, "sum", SUN),
        "일조_1_3월": (T(y, 1, 1), T(y, 3, 31), "sum", SUN),
        "누적온도_5도_2월말": (T(y, 2, 1), fe, "gdd", TEMP),
        "누적온도_5도_3월15일": (T(y, 2, 1), T(y, 3, 15), "gdd", TEMP),
        "누적온도_5도_3월말": (T(y, 2, 1), T(y, 3, 31), "gdd", TEMP),
    }


FEATURES = list(windows(2020))
USABLE = {
    "2월 말": list(C.USABLE_FEB_END),
    "3월 15일": list(C.USABLE_FEB_END) + ["누적온도_5도_3월15일"],
    "3월 31일": list(FEATURES),
}


def ref_date(name, y):
    if name == "2월 말":
        return feb_end(y)
    m, d = C.REF_DATES[name]
    return pd.Timestamp(y, m, d)


# 기준일의 날짜숫자를 돌려준다(1월 1일이 1)
def ref_doy(name, y):
    return int(ref_date(name, y).dayofyear)


# 일자료 읽기

# 연결 지점 10곳의 일자료와 그 출처를 돌려준다
# 열은 지점 일시 평균기온 일강수량 일조시간 다섯 개다
def load_daily():
    ids = C.stations()["기상지점"].unique().tolist()
    if C.WEATHER.exists():
        d = pd.read_csv(C.WEATHER, usecols=["지점", "일시", TEMP, RAIN, SUN], parse_dates=["일시"])
        src = "정제본(weather_daily.csv)"
    else:
        d = pd.read_csv(C.PUBLIC_TA, usecols=["지점", "일시", TEMP], parse_dates=["일시"])
        d[RAIN], d[SUN] = np.nan, np.nan
        src = "공개 기온(kma_ta_daily.csv). 강수와 일조는 비어 있다"
    d = d[d["지점"].isin(ids)].sort_values(["지점", "일시"]).reset_index(drop=True)
    return d, src


# 대구 새 자리 임시 지점 176의 하루 평균기온과 그 출처를 돌려준다
# 정제본에 있으면 정제본 값을 쓴다
# 없으면 공개 자료 파일을 쓰고 둘 다 없으면 None을 돌려준다
def load_176():
    if C.WEATHER.exists():
        w = pd.read_csv(C.WEATHER, usecols=["지점", "일시", TEMP], parse_dates=["일시"])
        w = w[w["지점"] == C.DAEGU["new"]].dropna()
        if len(w):
            return w.drop_duplicates("일시").set_index("일시")[TEMP], "정제본"
    if C.PUBLIC_TA176.exists():
        n = pd.read_csv(C.PUBLIC_TA176, usecols=["일시", TEMP], parse_dates=["일시"]).dropna()
        return n.drop_duplicates("일시").set_index("일시")[TEMP], "공개 자료 파일"
    return None, None


# 대구 143의 하루 평균기온을 새 자리(효목동) 기준으로 맞춘다
# 2013년 10월 10일부터 2015년 6월 11일까지는 새 자리 임시 지점 176의 값을 쓴다
# 이 기간에 176 값이 없는 날은 아래에 적은 달별 차이로 채운다
# 2013년 10월 9일까지는 옛 자리 값에 달별 평균 차이(새 자리 값에서 옛 자리 값을 뺀 것)를 더한다
# 이 차이는 두 자리가 함께 잰 날로 구한다
# 2015년 6월 12일부터는 143 기록이 이미 새 자리 값이라 그대로 둔다
def daegu_adjust(daily):
    D = C.DAEGU
    T0, T1 = pd.Timestamp(D["T0"]), pd.Timestamp(D["T1"])
    out = daily.copy()
    m = (out["지점"] == D["old"]).to_numpy()
    old = out.loc[m].set_index("일시")[TEMP]
    info = {"고친_지점": D["old"], "옛자리_마지막날": D["T1"]}
    new, src176 = load_176()
    if new is not None:
        both = pd.concat([old, new], axis=1, keys=["옛", "새"]).loc[T0:T1].dropna()
        diff = both["새"] - both["옛"]
        mdiff = diff.groupby(diff.index.month).mean()
        const = pd.Series(D["month_diff"])
        info.update({"방법": f"176 지점 값({src176}) + 겹친 날의 달별 차이", "겹친_날": int(len(both)),
                     "평균_차이": round(float(diff.mean()), 3), "차이_표준편차": round(float(diff.std(ddof=1)), 3),
                     "설정값과_달별차이_최대": round(float((mdiff - const.reindex(mdiff.index)).abs().max()), 3)})
        if len(mdiff) < 12:  # 겹친 날에 빠진 달이 있으면 설정값으로 채운다
            mdiff = mdiff.reindex(range(1, 13)).fillna(const)
    else:
        mdiff = pd.Series(D["month_diff"])
        info["방법"] = "설정의 달별 차이(176 파일 없음). 겹친 기간도 달별 차이로 맞춘다"
    info["달별_차이"] = {int(k): round(float(v), 2) for k, v in mdiff.items()}
    idx = old.index
    shifted = old + pd.Series(idx.month, index=idx).map(mdiff).to_numpy()
    adj = old.copy()
    pre = idx < T0
    mid = (idx >= T0) & (idx <= T1)
    adj[pre] = shifted[pre]
    if new is not None:
        v = new.reindex(idx)
        adj[mid] = v[mid].fillna(shifted[mid])
        info["겹친기간_176없어_채운날"] = int(v[mid].isna().sum())
    else:
        adj[mid] = shifted[mid]
    out.loc[m, TEMP] = adj.to_numpy()
    info["바꾼_날"] = int(((adj - old).abs() > 1e-9).sum() + (old.isna() & adj.notna()).sum())
    return out, info


# 변수 계산

def _station_frame(daily, sid, start, end):
    g = daily[daily["지점"] == sid].drop_duplicates("일시").set_index("일시").sort_index()
    return g.reindex(pd.date_range(start, end, freq="D"))[[TEMP, RAIN, SUN]]


# 한 지점 한 해의 변수를 계산한다
# has는 원자료 열마다 값이 하나라도 있는지 적은 것이다
# 변수 값 묶음과 가장 큰 결측 비율을 돌려준다
def year_features(g, y, has):
    row, worst = {}, 0.0
    for k, (a, b, how, col) in windows(y).items():
        if not has[col]:
            row[k] = np.nan
            continue
        x = g.loc[a:b, col]
        share = float(x.isna().mean()) if len(x) else 1.0
        worst = max(worst, share)
        if share > C.MISSING_MAX:
            row[k] = np.nan
        elif how == "mean":
            row[k] = round(float(x.mean()), 3)
        elif how == "sum":
            row[k] = round(float(x.sum()), 3)
        else:
            row[k] = round(float((x - C.GDD_BASE).clip(lower=0).sum()), 3)
    return row, round(worst, 3)


# 수목원 x 연도 변수 표를 만든다(10곳 x 17해)
def site_year_table(daily, years=None, adjusted=False):
    years = years or C.YEARS
    st = C.stations()
    start, end = f"{min(years) - 1}-12-01", f"{max(years)}-12-31"
    caution = set(C.WEATHER_CAUTION)
    rows = []
    for _, r in st.iterrows():
        g = _station_frame(daily, r["기상지점"], start, end)
        has = {c: bool(g[c].notna().any()) for c in (TEMP, RAIN, SUN)}
        for y in years:
            vals, worst = year_features(g, y, has)
            rows.append({"수목원": r["수목원"], "기상지점": r["기상지점"], "기상지점명": r["기상지점명"], "연도": y,
                         **vals, "결측비율_최대": worst,
                         "대구보정": bool(adjusted and r["수목원"] == C.DAEGU["site"] and y <= 2015),
                         "기상주의": (r["수목원"], y) in caution})
    return pd.DataFrame(rows)


# 누출 점검이다
# 기준일 뒤의 일자료를 999로 바꿔도 그 기준일에 쓰는 변수가 그대로인지 본다
# 어긋난 칸 수를 돌려준다
def leak_test_999(daily, years=None):
    years = years or C.YEARS
    st = C.stations()
    start, end = f"{min(years) - 1}-12-01", f"{max(years)}-12-31"
    bad = 0
    for _, r in st.iterrows():
        g = _station_frame(daily, r["기상지점"], start, end)
        has = {c: bool(g[c].notna().any()) for c in (TEMP, RAIN, SUN)}
        for y in years:
            base, _ = year_features(g, y, has)
            for name, cols in USABLE.items():
                g2 = g.copy()
                g2.loc[g2.index > ref_date(name, y)] = 999.0
                v2, _ = year_features(g2, y, has)
                for c in cols:
                    a, b = base[c], v2[c]
                    if not ((pd.isna(a) and pd.isna(b)) or a == b):
                        bad += 1
    return bad
