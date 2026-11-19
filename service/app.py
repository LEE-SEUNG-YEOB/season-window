"""Season Window Streamlit demonstration."""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
import season_functions as season

st.set_page_config(page_title="Season Window")
st.title("Season Window")

sites = season.list_sites()
if sites.empty:
    st.error("수목원 지점 자료를 찾을 수 없습니다.")
    st.stop()

points = pdk.Layer("ScatterplotLayer", sites, id="arboreta", pickable=True,
                   get_position=["경도", "위도"], get_fill_color=[36, 107, 80], get_radius=12000)
view = pdk.ViewState(latitude=sites["위도"].mean(), longitude=sites["경도"].mean(), zoom=5)
selection = st.pydeck_chart(pdk.Deck(layers=[points], initial_view_state=view,
                                     tooltip={"text": "{수목원}"}),
                            on_select="rerun", selection_mode="single-object")
selected = selection.selection.objects.get("arboreta", [])
map_site = selected[0]["수목원"] if selected else None
if map_site != st.session_state.get("map_site"):
    st.session_state.map_site = map_site
    if map_site in sites["수목원"].values:
        st.session_state.site = map_site
site = st.selectbox("수목원 전시원", sites["수목원"], key="site")

if not season.FINAL:
    st.warning("정제본 서비스 테이블이 필요합니다. 현재 자료는 집계표 미리 보기입니다.")
    st.stop()

start_default = dt.date(2027, 4, 3)
min_date, max_date = dt.date(season.YEAR_LO, 1, 1), dt.date(season.YEAR_HI, 12, 31)
page = st.radio("기능", ["개화·단풍 예상", "축제 기간 설계", "수종 추천"], horizontal=True)


def show_status(result):
    if not result["ok"]:
        st.info(result["message"])
        return False
    for warning in result["warnings"]:
        st.caption(warning)
    return True


if page == "개화·단풍 예상":
    phenomenon = st.radio("현상", ["개화", "단풍"], horizontal=True)
    species = st.selectbox("수종", season.list_species(site, phenomenon)["종"])
    year = st.number_input("대상 연도", season.YEAR_LO, season.YEAR_HI, 2027)
    if species:
        result = season.predict_date(site, species, phenomenon, year, lead="2월 말")
        if show_status(result):
            st.metric(result["stage"] + " 예상일", result["pred_date"],
                      f'{result["diff_from_mean_days"]:+g}일 · 관측 평균 대비')
            interval = result["interval"]
            st.write(f'{interval["level"]:.0%} 범위 · {interval["low_date"]} ~ {interval["high_date"]}')
            st.caption(f'{result["method"]} · 기준 관측 {result["n_years"]}년 · {result["lead"]}')
            history = result["history"].rename(columns={"날짜숫자": "과거 관측일"})[["연도", "과거 관측일"]]
            estimate = pd.DataFrame({"연도": [year], "예상일": [result["pred_doy"]]})
            st.scatter_chart(history.merge(estimate, on="연도", how="outer").sort_values("연도"),
                          x="연도", y=["과거 관측일", "예상일"], y_label="연중 일자")
            st.caption("연중 일자는 1월 1일을 1로 세는 값입니다.")
elif page == "축제 기간 설계":
    species = st.selectbox("수종", season.list_species(site)["종"])
    start = st.date_input("축제 시작일", start_default, min_date, max_date)
    days = st.number_input("축제 일수", 1, 60, 9)
    if species:
        result = season.festival_overlap(site, species, start, days, max_days=max(30, days))
        if show_status(result):
            st.metric("겹침 확률(분포 근사)", f'{result["prob"]:.1%}')
            if result["past"]:
                past = result["past"]
                st.metric("과거 실제 겹침", f'{past["n_years"]}해 중 {past["n_overlap"]}해')
            best = result["best"]
            st.write(f'같은 일수의 대안 · {best["start_date"]} ~ {best["end_date"]} ({best["prob"]:.1%})')
            st.caption(f'개화시작~낙화 길이 · {result["bloom_length_days"]:g}일 ({result["bloom_length_source"]})')
            st.line_chart(result["by_days"], x="일수", y=["확률", "최적기간_확률"])
else:
    phenomenon = st.radio("현상", ["개화", "단풍"], horizontal=True)
    start = st.date_input("희망 시작일", start_default, min_date, max_date)
    end = st.date_input("희망 종료일", start_default + dt.timedelta(days=8), min_date, max_date)
    result = season.recommend_species(site, start, end, phenomenon)
    if show_status(result):
        st.dataframe(result["ranking"], hide_index=True)
