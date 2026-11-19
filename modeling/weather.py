# 1단계 일자료로 수목원 x 연도 기상 변수 표 생성과 누출 점검

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import features as F
import mconfig as C

def main():
    print("자료판:", C.mode())
    daily, src = F.weather()
    print("기상 일자료:", src, "|", daily["지점"].nunique(), "지점", len(daily), "줄",
          daily["일시"].min().date(), "~", daily["일시"].max().date())

    # A의 정제본과 공개 기온이 둘 다 있으면 평균기온이 같은지 확인
    if C.WEATHER.exists() and C.PUBLIC_TA.exists():
        pub = pd.read_csv(C.PUBLIC_TA, usecols=["지점", "일시", F.TEMP], parse_dates=["일시"])
        j = daily.merge(pub, on=["지점", "일시"], suffixes=("", "_공개")).dropna(subset=[F.TEMP, F.TEMP + "_공개"])
        gap = (j[F.TEMP] - j[F.TEMP + "_공개"]).abs()
        print(f"정제본과 공개 기온 비교: 겹친 {len(j)}일, 0.05도 넘게 다른 날 {(gap > 0.05).sum()}일, 최대 {gap.max():.2f}도")

    raw = F.table(daily, adjusted=False)
    adj_daily, info = F.correction(daily)
    adj = F.table(adj_daily, adjusted=True)
    print("대구 보정:", info)

    main_tab, other = (adj, raw) if C.DAEGU_ADJUST else (raw, adj)
    out = C.folder("features")
    main_path = out / C.FEATURES.name
    other_path = out / (C.FEATURES_RAW.name if C.DAEGU_ADJUST else "weather_features_daegu_adj.csv")
    main_tab.to_csv(main_path, index=False, encoding="utf-8-sig")
    other.to_csv(other_path, index=False, encoding="utf-8-sig")
    print("저장:", main_path.relative_to(C.ROOT), "/", other_path.relative_to(C.ROOT), "|", main_tab.shape)

    # 결과 점검
    ok = True
    if len(main_tab) != len(C.stations()) * len(C.YEARS):
        print("!! 줄 수가 수목원 x 연도와 다르다"); ok = False
    print("창 안 결측 비율 최대:", main_tab["결측비율_최대"].max(),
          "| 20% 규칙으로 빈칸이 된 기온 칸:", int(main_tab[[c for c in F.FEATURES if c.startswith("기온")]].isna().sum().sum()))
    top = main_tab.nlargest(3, "결측비율_최대")[["수목원", "기상지점명", "연도", "결측비율_최대"]]
    print(top.to_string(index=False))
    zero = (main_tab["누적온도_5도_2월말"] == 0).groupby(main_tab["기상지점명"]).sum()
    print("누적온도_5도_2월말이 0인 해:", zero[zero > 0].sort_values(ascending=False).to_dict())
    d = (adj.set_index(["수목원", "연도"])[F.FEATURES] - raw.set_index(["수목원", "연도"])[F.FEATURES]).abs()
    site = d.groupby(level=0).max().max(axis=1)
    late = d.xs(C.DAEGU["site"], level=0).loc[2016:].max().max()
    print(f"보정이 바꾼 곳: {site[site > 0].index.tolist()} | 대구 2016년 이후 바뀐 폭 최대 {late:.3f} (0이어야 함)")
    if site.drop(C.DAEGU["site"]).max() > 0 or late > 0:
        print("!! 보정이 대구 2015년 이전 밖을 건드렸다"); ok = False
    print("기상주의 줄:", main_tab.loc[main_tab["기상주의"], ["수목원", "연도"]].values.tolist())

    bad = F.leak(adj_daily)
    print("누출 점검(기준일 뒤 999):", "통과" if bad == 0 else f"실패 {bad}칸")
    ok = ok and bad == 0
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
