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


def species_options(site, phenomenon, column):
    """계산할 수 있는 종만 고른다(관측 8년 미만인 종은 목록에서 뺀다)."""
    table = season.list_species(site, phenomenon)
    if not isinstance(table, pd.DataFrame):
        return []
    return table.loc[table[column], "종"].tolist()


def show_status(result):
    if not result["ok"]:
        st.info(result["message"])
        return False
    for warning in result["warnings"]:
        st.caption(warning)
    return True


if page == "개화·단풍 예상":
    phenomenon = st.radio("현상", ["개화", "단풍"], horizontal=True)
    options = species_options(site, phenomenon, "예측가능")
    if not options:
        st.info(f"{site}에는 {phenomenon} 날짜를 8년 이상 관측한 종이 없습니다.")
    species = st.selectbox("수종", options) if options else None
    year = st.number_input("대상 연도", season.YEAR_LO, season.YEAR_HI, 2027)
    if species:
        # 2월 기온 모델(M1)은 검증에서 기각되어 채택된 계획 값(과거 평균 + 최근 5년 보정)을 쓴다
        result = season.predict_date(site, species, phenomenon, year, lead="계획")
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
    options = species_options(site, "개화", "추천가능")
    if not options:
        st.info(f"{site}에는 개화시작과 낙화를 함께 8년 이상 관측한 종이 없습니다.")
    species = st.selectbox("수종", options) if options else None
    start = st.date_input("축제 시작일", start_default, min_date, max_date)
    days = st.number_input("축제 일수", 1, 60, 9)
    if species:
        result = season.festival_overlap(site, species, start, days, max_days=max(30, days))
        if show_status(result):
            # 대표값은 지난 해 실제로 겹친 비율(기능 3의 적합도와 같은 기준)
            headline = result["headline"]
            if headline["basis"] == "과거 해":
                st.metric("지난 해 실제 겹침", headline["text"])
                label = "최근 5년 경향을 반영한 확률(근사)" if season.USE_M0R.get("개화") else "겹칠 확률(근사)"
                st.metric(label, f'{result["prob"]:.0%}')
            else:
                st.metric("겹칠 확률(근사)", f'{result["prob"]:.0%}')
            best = result["best"]
            st.write(f'같은 일수에서 확률이 가장 높은 기간(근사) · {best["start_date"]} ~ {best["end_date"]} '
                     f'({best["prob"]:.0%})')
            st.caption(f'개화시작~낙화 길이 · {result["bloom_length_days"]:g}일 ({result["bloom_length_source"]})')
            st.line_chart(result["by_days"], x="일수", y=["확률", "최적기간_확률"])
else:
    # 단풍은 시기의 시작과 끝을 어느 단계로 볼지 정해지지 않아 기능 함수가 아직 계산하지 않는다
    st.caption("수종 추천은 봄꽃(개화)만 지원합니다. 단풍은 단풍 시기의 정의가 정해진 뒤 추가합니다.")
    start = st.date_input("희망 시작일", start_default, min_date, max_date)
    end = st.date_input("희망 종료일", start_default + dt.timedelta(days=8), min_date, max_date)
    result = season.recommend_species(site, start, end, "개화")
    if show_status(result):
        st.dataframe(result["ranking"], hide_index=True)
