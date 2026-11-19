# 기능 1~3 함수 파일이 설계서 7절의 약속을 지키는지 점검한다
# 저장소 맨 위에서 python3로 modeling 폴더의 이 파일을 돌린다
#
# 집계표 판과 정제본 판 어느 쪽에서도 돈다
# 맨 앞에 지금 판을 찍는다
#
# 보는 것은 아래와 같다
# 반환 키와 표의 열을 본다
# 범위의 순서와 확률의 범위를 본다
# 정렬을 본다
# 기능 2와 3이 같은 값을 내는지 본다
# 오류 코드를 본다
# 예외가 없는지 본다
#
# 정제본 판이면 10절의 맞춰 보기도 한다
# 지난 해 세기를 따로 다시 세어 보고 A의 하루 적중률과 견준다
# 하나라도 틀리면 목록을 찍고 종료 코드 1로 끝난다

import datetime as dt
import sys
import unicodedata
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import season_functions as F

KEYS = {
    "predict_date": ["ok", "site", "species", "phenomenon", "stage", "year", "lead", "method", "pred_doy", "pred_date",
                     "interval", "mean_doy", "diff_from_mean_days", "sd_days", "n_years", "actual_doy", "history",
                     "scenario", "warnings", "source"],
    "festival_overlap": ["ok", "site", "species", "phenomenon", "lead", "start_date", "days", "end_date",
                         "min_overlap_days", "bloom_length_days", "bloom_length_source", "prob", "headline", "grade",
                         "expected_overlap_days", "best", "by_days", "by_start", "past", "scenario", "basis",
                         "n_years", "warnings", "source"],
    "recommend_species": ["ok", "site", "phenomenon", "start_date", "end_date", "min_overlap_days", "n_candidates",
                          "fit_method", "ranking", "excluded", "warnings", "source"],
}
INTERVAL = ["level", "low_doy", "high_doy", "low_date", "high_date"]
WINDOW = ["start_doy", "start_date", "end_doy", "end_date"]
HEADLINE = ["value", "basis", "text"]
BEST = ["start_date", "end_date", "prob"]
SCENARIO = ["temp_anom", "window", "slope_days_per_degree", "shift_days"]
PAST = ["n_years", "n_overlap", "rate", "years_overlap", "years_miss"]
BY_DAYS = ["일수", "확률", "최적기간_확률", "기대겹침일수"]
BY_START = ["시작일_날짜숫자", "시작일", "확률"]
RANKING = ["순위", "종", "적합도", "안정성점수", "관측연수", "평균_개화시작", "평균_낙화", "변동폭_표준편차",
           "개화기간_중앙값", "등급", "겹친해수", "기대겹침일수"]
EXCLUDED = ["종", "사유"]
HISTORY = ["연도", "날짜숫자", "날짜", "평균대비_일"]
SITE_COLS = ["수목원", "위도", "경도", "관측시작연도", "관측종료연도"]
SPECIES_COLS = ["종", "관측연수", "예측가능", "추천가능"]
WINDOWS = [("2027-03-25", "2027-04-03"), ("2027-04-10", "2027-04-16"), ("2027-05-10", "2027-05-12")]

N_OK, FAILS, NOTES = [0], [], []


def check(cond, what):
    if cond:
        N_OK[0] += 1
    else:
        FAILS.append(what)
    return bool(cond)


# 함수를 부른다
# 예외가 나면 실패로 적고 None을 돌려준다
def call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        FAILS.append(f"예외 {fn.__name__}{a}: {type(e).__name__}: {e}")
        return None


def has(d, keys, what):
    miss = [k for k in keys if not isinstance(d, dict) or k not in d]
    return check(not miss, f"{what}: 빠진 키 {miss}")


def cols(df, want, what, exact=False):
    ok = isinstance(df, pd.DataFrame) and (list(df.columns) == want if exact else all(c in df.columns for c in want))
    return check(ok, f"{what}: 열이 약속과 다름 {list(df.columns) if isinstance(df, pd.DataFrame) else type(df)}")


def code(r, want, what):
    got = r.get("code") if isinstance(r, dict) else None
    ok = isinstance(r, dict) and r.get("ok") is False and got == want and isinstance(r.get("message"), str) \
        and "source" in r and isinstance(r.get("warnings"), list)
    return check(ok, f"{what}: {want}를 기대했는데 {got}")


def date_of(year, k):
    return dt.date(year, 1, 1) + dt.timedelta(days=int(k) - 1)


# 지난 해 세기를 함수와 따로 다시 센다
# 축제는 달과 날로 맞추고 날마다 꽃이 피어 있는지 본다
def count_md(site, sp, start, days, m=1, length=None):
    P = F._B[(F._B["수목원"] == site) & (F._B["종"] == sp)]
    n = 0
    for y, on, off in zip(P["연도"].astype(int), P["개화시작"], P["낙화"]):
        try:
            s = dt.date(y, start.month, start.day)
        except ValueError:
            s = dt.date(y, 2, 28)
        s = s.timetuple().tm_yday
        end = on + length if length is not None else off
        n += sum(on <= k <= end for k in range(s, s + days)) >= m
    return n


def t_lists():
    S = call(F.list_sites)
    cols(S, SITE_COLS, "list_sites", exact=True)
    check(S is not None and len(S) == 10, "list_sites: 10곳이 아님")
    for site in F.SITES:
        for phen, pool in (("개화", F.SPRING_SPECIES), ("단풍", F.AUTUMN_SPECIES)):
            L = call(F.list_species, site, phen)
            if cols(L, SPECIES_COLS, f"list_species {site} {phen}", exact=True):
                check(L["종"].tolist() == pool, f"list_species {site} {phen}: 종 목록이 다름")
                check(phen == "개화" or not L["추천가능"].any(), f"list_species {site} 단풍: 추천가능이 참")
    code(call(F.list_species, "없는곳"), "UNKNOWN_SITE", "list_species 없는 곳")
    code(call(F.list_species, F.SITES[0], "낙엽"), "UNKNOWN_PHENOMENON", "list_species 없는 현상")


def t_predict():
    for site in F.SITES:
        for phen, pool in (("개화", F.SPRING_SPECIES), ("단풍", F.AUTUMN_SPECIES)):
            L = F.list_species(site, phen).set_index("종")
            for sp in pool:
                what = f"predict_date {site} {sp}"
                r = call(F.predict_date, site, sp, phen)
                if r is None:
                    continue
                check(r.get("ok") == bool(L.loc[sp, "예측가능"]), f"{what}: ok와 list_species의 예측가능이 다름")
                if not r.get("ok"):
                    check(r.get("code") in ("NO_DATA", "TOO_FEW_YEARS") and "n_years" in r,
                          f"{what}: 실패 코드 {r.get('code')}")
                    continue
                has(r, KEYS["predict_date"], what)
                has(r["interval"], INTERVAL, what + " interval")
                iv = r["interval"]
                check(iv["low_doy"] <= r["pred_doy"] <= iv["high_doy"]
                      and iv["low_date"] <= r["pred_date"] <= iv["high_date"], f"{what}: 범위 순서")
                check(r["n_years"] >= F.MIN_YEARS and r["source"] == F.SOURCE and isinstance(r["warnings"], list),
                      f"{what}: n_years, source, warnings")
                check(r["scenario"] is None and r["actual_doy"] is None and r["lead"] == "계획",
                      f"{what}: 2027년 계획인데 scenario나 actual_doy가 있음")
                want = round(F.RECENT5[phen], 1) if F.USE_M0R.get(phen) else 0.0
                check(abs(r["diff_from_mean_days"] - want) < 0.051, f"{what}: 최근 5년 보정값이 판정과 다름")
                widths = [F.predict_date(site, sp, phen, level=lv)["interval"] for lv in (0.5, 0.8, 0.9)]
                widths = [q["high_doy"] - q["low_doy"] for q in widths]
                check(widths[0] < widths[1] < widths[2], f"{what}: 범위 수준이 높을수록 넓어야 함 {widths}")
                a, b = F._row(site, sp, "개화시작"), F._row(site, sp, "낙화90~100%")
                want_bw = phen == "개화" and a is not None and b is not None \
                    and min(a["관측연수"], b["관측연수"]) >= F.MIN_YEARS
                check(("bloom_window" in r) == want_bw, f"{what}: bloom_window 유무가 규칙과 다름")
                if "bloom_window" in r:
                    bw = r["bloom_window"]
                    has(bw, WINDOW, what + " bloom_window")
                    check(bw["start_doy"] <= bw["end_doy"], f"{what}: bloom_window 순서")
                h = r["history"]
                if not F.FINAL:
                    check(h is None, f"{what}: 집계표 판인데 history가 있음")
                    continue
                if cols(h, HISTORY, what + " history", exact=True):
                    check(len(h) == int(F._row(site, sp, r["stage"])["관측연수"]), f"{what}: history 줄 수")
                    y0, v0 = int(h["연도"].iloc[-1]), float(h["날짜숫자"].iloc[-1])
                    q = F.predict_date(site, sp, phen, year=y0)
                    check(q["actual_doy"] == v0 and any("지난 해를 골랐습니다" in w for w in q["warnings"]),
                          f"{what}: 지난 해 {y0}의 actual_doy나 경고")


def t_scenario(site, sp, au):
    base = F.predict_date(site, sp)
    bw = base["interval"]["high_doy"] - base["interval"]["low_doy"]
    for x in (-2, -1.0, 0.5, 2.0):
        r = F.predict_date(site, sp, temp_scenario=x)
        what = f"predict_date 기온 가정 {x}"
        if has(r.get("scenario"), SCENARIO, what):
            check(r["scenario"]["shift_days"] == round(F.SLOPE * x, 1), f"{what}: shift_days")
        check(abs((r["pred_doy"] - base["pred_doy"]) - F.SLOPE * x) <= 0.11, f"{what}: 예상 날짜가 기울기만큼 옮겨지지 않음")
        check(abs((r["interval"]["high_doy"] - r["interval"]["low_doy"]) - bw) <= 0.11, f"{what}: 범위 폭이 바뀜")
        check(r["lead"] == "시나리오", f"{what}: lead")
    r = F.predict_date(site, au, "단풍", temp_scenario=1.0)
    check(r.get("ok") is not True or (r["scenario"] is None and any("단풍에는 기온 가정" in w for w in r["warnings"])),
          "predict_date 단풍 기온 가정: 반영하지 않는다는 경고가 없음")
    f = F.festival_overlap(site, sp, "2027-04-01", 7, temp_scenario=1.0)
    if has(f.get("scenario"), SCENARIO, "festival_overlap 기온 가정"):
        check(f["lead"] == "시나리오" and f["headline"]["basis"] == "분포" and f["basis"].endswith("+ 기온 가정"),
              "festival_overlap 기온 가정: lead, headline, basis")


def t_lead(site, sp):
    r = F.predict_date(site, sp, lead="2월 말")
    if F.FINAL and F.USE_M1:
        check(r["lead"] == "계획" and any("2월 기온이 아직 없어" in w for w in r["warnings"]),
              "predict_date 2월 말(입력 없는 해): 계획 값과 경고")
        y = int(F._M1X["연도"].max())
        q = F.predict_date(site, sp, year=y, lead="2월 말")
        check(q["method"] == "2월 말 예측(M1)" and q["lead"] == "2월 말", f"predict_date 2월 말({y}): M1을 쓰지 않음")
        lo, hi = q["interval"]["low_doy"], q["interval"]["high_doy"]
        check(lo <= q["pred_doy"] <= hi, f"predict_date 2월 말({y}): 범위 순서")
    else:
        check(r["lead"] == "계획" and r["method"] in ("기준(과거 평균)", "최근 5년 보정") and r["warnings"],
              "predict_date 2월 말: M1을 안 쓰는데 계획 값이 아니거나 경고가 없음")
    q = F.predict_date(site, sp, lead="엉뚱한값")
    check(q.get("ok") and any("lead는" in w for w in q["warnings"]), "predict_date lead 오입력: 경고가 없음")
    # 시나리오는 기온 가정이 있어야 쓰고 없으면 계획 값과 경고를 낸다
    need = "temp_scenario"
    q = F.predict_date(site, sp, lead="시나리오")
    check(q.get("ok") and q["lead"] == "계획" and any(need in w for w in q["warnings"]),
          "predict_date 기온 가정 없는 시나리오: 계획 값과 경고")
    q = F.predict_date(site, sp, lead="시나리오", temp_scenario=1.0)
    check(q.get("ok") and q["lead"] == "시나리오" and not any(need in w or "lead는" in w for w in q["warnings"]),
          "predict_date 기온 가정 있는 시나리오: lead나 경고")
    q = F.predict_date(site, sp)
    check(q.get("ok") and not any(need in w or "lead는" in w for w in q["warnings"]), "predict_date 계획: lead 경고가 붙음")
    FO = F.festival_overlap
    for ld, want in (("시나리오", need), ("2월 말", "2월 말 예측을 쓰지 않아"), ("엉뚱한값", "lead는")):
        f = FO(site, sp, "2027-04-01", 7, lead=ld)
        check(f.get("ok") and f["lead"] == "계획" and any(want in w for w in f["warnings"]),
              f"festival_overlap lead={ld}: 계획 값과 경고")
    for ld, x, want in (("시나리오", 1.0, "시나리오"), ("계획", None, "계획")):
        f = FO(site, sp, "2027-04-01", 7, lead=ld, temp_scenario=x)
        check(f.get("ok") and f["lead"] == want and not any(need in w or "lead는" in w for w in f["warnings"]),
              f"festival_overlap lead={ld}: lead나 경고")


def t_predict_errors(site, sp, au):
    P = F.predict_date
    for y in (2008, 2031, "2027", True, 2027.0, None):
        code(call(P, site, sp, year=y), "BAD_YEAR", f"predict_date year={y!r}")
    for lv in (0.7, "0.8", True, None, [0.8]):
        code(call(P, site, sp, level=lv), "BAD_LEVEL", f"predict_date level={lv!r}")
    for x in (3, -2.5, float("nan"), float("inf"), "1", True, [1]):
        code(call(P, site, sp, temp_scenario=x), "BAD_SCENARIO", f"predict_date temp_scenario={x!r}")
    code(call(P, "없는곳", sp), "UNKNOWN_SITE", "predict_date 없는 곳")
    code(call(P, ["국립수목원"], sp), "UNKNOWN_SITE", "predict_date 목록으로 넣은 곳")
    code(call(P, site, "장미"), "UNKNOWN_SPECIES", "predict_date 없는 종")
    code(call(P, site, au), "UNKNOWN_SPECIES", "predict_date 개화에 단풍 종")
    code(call(P, site, sp, "낙엽"), "UNKNOWN_PHENOMENON", "predict_date 없는 현상")


def t_overlap():
    for site in F.SITES:
        for sp in F.SPRING_SPECIES:
            what = f"festival_overlap {site} {sp}"
            o = F._row(site, sp, "개화시작")
            start = date_of(2027, round(float(o["평균"]))) if o is not None else dt.date(2027, 4, 1)
            r = call(F.festival_overlap, site, sp, start.isoformat(), 7)
            if r is None:
                continue
            if not r.get("ok"):
                check(r.get("code") in ("NO_DATA", "TOO_FEW_YEARS"), f"{what}: 실패 코드 {r.get('code')}")
                continue
            has(r, KEYS["festival_overlap"], what)
            has(r["headline"], HEADLINE, what + " headline")
            has(r["best"], BEST, what + " best")
            check(0 <= r["prob"] <= 1 and 0 <= r["best"]["prob"] <= 1, f"{what}: 확률이 0~1 밖")
            check(r["best"]["prob"] >= r["prob"], f"{what}: 가장 좋은 기간의 확률이 입력한 기간보다 낮음")
            check(r["end_date"] == (start + dt.timedelta(days=6)).isoformat(), f"{what}: end_date")
            bd, bs = r["by_days"], r["by_start"]
            if cols(bd, BY_DAYS, what + " by_days", exact=True):
                check(len(bd) == 30 and bd["확률"].is_monotonic_increasing and bd["기대겹침일수"].is_monotonic_increasing
                      and (bd["최적기간_확률"] >= bd["확률"]).all(), f"{what}: by_days 곡선 모양")
                check(abs(bd.loc[bd["일수"] == 7, "확률"].iloc[0] - r["prob"]) < 1e-9, f"{what}: by_days와 prob가 다름")
            if cols(bs, BY_START, what + " by_start", exact=True):
                check(bs["확률"].between(0, 1).all(), f"{what}: by_start 확률 범위")
            check(r["grade"] == F.grade(r["headline"]["value"]), f"{what}: grade가 headline과 다름")
            check(0 <= r["expected_overlap_days"] <= r["days"], f"{what}: 기대 겹침 일수 범위")
            m3 = F.festival_overlap(site, sp, start, 7, min_overlap_days=3)
            check(m3["prob"] <= r["prob"], f"{what}: 겹침 기준을 3일로 올렸는데 확률이 올라감")
            check(F.festival_overlap(site, sp, start, 7)["prob"] == r["prob"], f"{what}: date와 글자 입력 결과가 다름")
            p = r["past"]
            if not F.FINAL:
                check(p is None and r["headline"]["basis"] == "분포", f"{what}: 집계표 판인데 past가 있음")
                continue
            if not has(p, PAST, what + " past"):
                continue
            yrs = sorted(F._B.loc[(F._B["수목원"] == site) & (F._B["종"] == sp), "연도"].astype(int))
            check(sorted(p["years_overlap"] + p["years_miss"]) == yrs and p["n_years"] == len(yrs)
                  and p["n_overlap"] == len(p["years_overlap"]), f"{what}: past의 해 목록")
            check(p["n_overlap"] == count_md(site, sp, start, 7), f"{what}: 따로 센 겹친 해 수와 다름")
            if p["n_years"] >= F.MIN_YEARS:
                check(r["headline"]["basis"] == "과거 해" and r["headline"]["value"] == p["rate"]
                      and r["headline"]["text"] == f"{p['n_years']}해 중 {p['n_overlap']}해 겹침",
                      f"{what}: headline이 past와 다름")
            L = F.festival_overlap(site, sp, start, 7, bloom_length=10)
            check(L["bloom_length_source"] == "직접 입력" and L["bloom_length_days"] == 10
                  and L["past"]["n_overlap"] == count_md(site, sp, start, 7, length=10),
                  f"{what}: 꽃 시기 길이를 넣었을 때 지난 해 세기")


def t_overlap_errors(site, sp, au):
    FO = F.festival_overlap
    for d in ("2027-13-01", "2027-02-30", "abc", None, ""):
        code(call(FO, site, sp, d, 7), "BAD_DATE", f"festival_overlap start_date={d!r}")
    for n in (0, 61, 2.5, True, "7", None):
        code(call(FO, site, sp, "2027-04-01", n), "BAD_DAYS", f"festival_overlap days={n!r}")
    for n in (0, 61, 2.5):
        code(call(FO, site, sp, "2027-04-01", 7, max_days=n), "BAD_DAYS", f"festival_overlap max_days={n!r}")
    for n in (0, 8, 1.5, True):
        code(call(FO, site, sp, "2027-04-01", 7, min_overlap_days=n), "BAD_DAYS",
             f"festival_overlap min_overlap_days={n!r}")
    for L in (0, 61, True, "10", float("nan")):
        code(call(FO, site, sp, "2027-04-01", 7, bloom_length=L), "BAD_LENGTH", f"festival_overlap bloom_length={L!r}")
    code(call(FO, site, sp, "2027-04-01", 7, temp_scenario=3), "BAD_SCENARIO", "festival_overlap temp_scenario=3")
    code(call(FO, site, au, "2027-10-20", 7, "단풍"), "NOT_SUPPORTED", "festival_overlap 단풍")
    code(call(FO, "없는곳", sp, "2027-04-01", 7), "UNKNOWN_SITE", "festival_overlap 없는 곳")
    code(call(FO, site, "장미", "2027-04-01", 7), "UNKNOWN_SPECIES", "festival_overlap 없는 종")
    r = call(FO, site, sp, "2028-02-29", 7)
    check(r is not None and r.get("ok"), "festival_overlap 2월 29일 입력이 실패함")
    # 축제 시작일의 해도 기능 1과 같은 범위만 받는다
    for d in ("2008-12-31", "2031-01-01", "9999-12-31", "0001-01-01", dt.date(9999, 12, 31)):
        code(call(FO, site, sp, d, 7), "BAD_DATE", f"festival_overlap 범위 밖 해 {d!r}")
    for d in ("2009-01-01", "2030-12-31"):
        r = call(FO, site, sp, d, 7)
        check(r is not None and r.get("ok"), f"festival_overlap 범위 끝 해 {d}가 실패함")


def t_recommend():
    for site in F.SITES:
        for a, b in WINDOWS:
            what = f"recommend_species {site} {a}~{b}"
            r = call(F.recommend_species, site, a, b)
            if r is None:
                continue
            if not r.get("ok"):
                code(r, "NO_CANDIDATE", what)
                check(isinstance(r.get("excluded"), pd.DataFrame), f"{what}: excluded가 없음")
                continue
            has(r, KEYS["recommend_species"], what)
            R, X = r["ranking"], r["excluded"]
            if not (cols(R, RANKING, what + " ranking") and cols(X, EXCLUDED, what + " excluded", exact=True)):
                continue
            check(R["순위"].tolist() == list(range(1, len(R) + 1)) and r["n_candidates"] == len(R), f"{what}: 순위 번호")
            key = list(zip(R["적합도"], R["안정성점수"].fillna(-1)))
            check(all(key[i] >= key[i + 1] for i in range(len(key) - 1)), f"{what}: 적합도와 안정성점수 순서")
            check(set(R["종"]).isdisjoint(X["종"]) and set(R["종"]) | set(X["종"]) == set(F.SPRING_SPECIES),
                  f"{what}: 순위와 제외 목록이 11종을 나누지 않음")
            want = {sp for sp in F.SPRING_SPECIES if F._hit(site, sp) is not None and "ok" not in F._onset(site, sp)}
            check(set(R["종"]) == want, f"{what}: 후보가 개화시작과 낙화 8년 이상 조합과 다름")
            check(R["적합도"].between(0, 1).all() and (R["등급"] == R["적합도"].map(F.grade)).all(), f"{what}: 적합도와 등급")
            check(R["적합도"].max() > 0, f"{what}: 적합도가 모두 0인데 ok")
            check(r["fit_method"] == ("지난 해 세기" if F.FINAL else "정규 근사(임시)"), f"{what}: fit_method")
            days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1
            for _, row in R.iterrows():
                fo = F.festival_overlap(site, row["종"], a, days)
                if F.FINAL:
                    p = fo["past"]
                    check(row["겹친해수"] == p["n_overlap"] and row["관측연수"] == p["n_years"]
                          and abs(row["적합도"] - p["n_overlap"] / p["n_years"]) <= 0.005 + 1e-9,
                          f"{what} {row['종']}: 기능 3의 적합도가 기능 2의 지난 해 세기와 다름")
                else:
                    check(abs(row["적합도"] - fo["prob"]) <= 0.0051 and row["겹친해수"] is None,
                          f"{what} {row['종']}: 기능 3의 적합도가 기능 2의 확률과 다름")
    one = call(F.recommend_species, F.SITES[0], "2027-04-10", "2027-04-10")
    check(one is not None and one.get("ok") is not None, "recommend_species 하루 창")


def t_recommend_errors(site):
    RS = F.recommend_species
    code(call(RS, site, "2027-04-10", "2027-04-01"), "BAD_DATE", "recommend_species 종료일이 시작일보다 앞")
    code(call(RS, site, "2027-12-25", "2028-01-05"), "BAD_DATE", "recommend_species 해가 다름")
    code(call(RS, site, "2027-02-30", "2027-03-05"), "BAD_DATE", "recommend_species 없는 날짜")
    code(call(RS, site, "2027-04-01", "2027-04-07", min_overlap_days=0), "BAD_DAYS", "recommend_species 겹침 0일")
    code(call(RS, site, "2027-04-01", "2027-04-07", min_overlap_days=8), "BAD_DAYS", "recommend_species 겹침 8일")
    code(call(RS, site, "2027-10-01", "2027-10-07", "단풍"), "NOT_SUPPORTED", "recommend_species 단풍")
    code(call(RS, "없는곳", "2027-04-01", "2027-04-07"), "UNKNOWN_SITE", "recommend_species 없는 곳")
    for a, b in (("2008-04-01", "2008-04-07"), ("2031-04-01", "2031-04-07"), ("9999-12-25", "9999-12-31")):
        code(call(RS, site, a, b), "BAD_DATE", f"recommend_species 범위 밖 해 {a}")
    # 꽃 철이 아닌 기간은 적합도 0인 순위표 대신 후보 없음으로 답한다
    for a, b in (("2027-12-01", "2027-12-07"), ("2027-01-10", "2027-01-20")):
        r = call(RS, site, a, b)
        if code(r, "NO_CANDIDATE", f"recommend_species 꽃 철이 아닌 기간 {a}"):
            cols(r.get("excluded"), EXCLUDED, f"recommend_species 꽃 철이 아닌 기간 {a} excluded", exact=True)
    saved = F._H
    F._H = F._H.iloc[0:0]
    try:
        r = call(RS, site, "2027-04-01", "2027-04-07")
    finally:
        F._H = saved
    if code(r, "NO_CANDIDATE", "recommend_species 후보 없음"):
        check(isinstance(r.get("excluded"), pd.DataFrame) and len(r["excluded"]) == len(F.SPRING_SPECIES),
              "recommend_species 후보 없음: excluded")


# 맥에서 복사한 한글(자모가 풀린 꼴)도 같은 이름으로 받는지 본다
def t_nfc(site, sp):
    a, b, k, ld = (unicodedata.normalize("NFD", x) for x in (site, sp, "개화", "시나리오"))
    check(a != site and k != "개화", "풀린 한글 점검: 바꾼 글자가 원래와 같음")
    L = call(F.list_species, a, k)
    check(isinstance(L, pd.DataFrame) and L["종"].tolist() == F.SPRING_SPECIES, "list_species 풀린 한글")
    r = call(F.predict_date, a, b, k, lead=ld, temp_scenario=1.0)
    check(isinstance(r, dict) and r.get("ok") and r["site"] == site and r["species"] == sp
          and not any("lead는" in w for w in r["warnings"]), "predict_date 풀린 한글")
    f = call(F.festival_overlap, a, b, "2027-04-01", 7, k)
    check(isinstance(f, dict) and f.get("ok") and f["site"] == site and f["species"] == sp
          and f["prob"] == F.festival_overlap(site, sp, "2027-04-01", 7)["prob"], "festival_overlap 풀린 한글")
    g = call(F.recommend_species, a, "2027-04-01", "2027-04-07", k)
    check(isinstance(g, dict) and g.get("ok") and g["site"] == site, "recommend_species 풀린 한글")


# 함수 안에서 예기치 못한 오류가 나도 예외 대신 내부 오류 코드를 돌려주는지 본다
def t_internal(site, sp):
    def boom(*a, **k):
        raise RuntimeError("점검용 오류")
    for name, fn, args in (("_row", F.predict_date, (site, sp)),
                           ("_onset", F.festival_overlap, (site, sp, "2027-04-01", 7)),
                           ("_onset", F.recommend_species, (site, "2027-04-01", "2027-04-07"))):
        saved = getattr(F, name)
        setattr(F, name, boom)
        try:
            r = call(fn, *args)
        finally:
            setattr(F, name, saved)
        code(r, "INTERNAL_ERROR", f"{fn.__name__} 내부 오류")


# 10절 맞춰 보기다
# 정제본 판에서만 한다
def t_against_a():
    if not F.FINAL:
        NOTES.append("집계표 판이라 10절 맞춰 보기(지난 해 세기와 A의 하루_적중률)는 건너뛴다.")
        return
    A = pd.read_csv(F.TABLES / "variability_site_species.csv")
    J = F._V.merge(A, on=["수목원", "종", "현상"], suffixes=("", "_A"))
    same = len(J) > 0 and (J["관측연수"] == J["관측연수_A"]).all() \
        and ((J["평균"] - J["평균_A"]).abs() < 0.01).all() and ((J["표준편차"] - J["표준편차_A"]).abs() < 0.01).all()
    Hh = pd.read_csv(F.TABLES / "eda_fixed_day_hit.csv")
    n = n_a = n_md = 0
    days = np.arange(1, 201)
    for _, h in Hh.iterrows():
        P = F._B[(F._B["수목원"] == h["수목원"]) & (F._B["종"] == h["종"])]
        if P.empty:
            continue
        n += 1
        on, off = P["개화시작"].to_numpy()[:, None], P["낙화"].to_numpy()[:, None]
        inb = (days >= on) & (days <= off)
        one, three = inb.mean(axis=0), (inb[:, :-2] & inb[:, 1:-1] & inb[:, 2:]).mean(axis=0)
        n_a += int(days[one.argmax()]) == h["최적고정일"] and round(one.max(), 3) == h["하루_적중률"] \
            and round(three.max(), 3) == h["사흘연속_적중률"]
        r = F.festival_overlap(h["수목원"], h["종"], date_of(2027, h["최적고정일"]), 1)
        p = r.get("past") if r.get("ok") else None
        n_md += bool(p) and abs(p["rate"] - h["하루_적중률"]) < 0.0006
    NOTES.append(f"8년 이상 조합: 함수 {len(F._H)}개 {int(F._H['관측연수'].sum())}해, A의 표 {len(Hh)}개 "
                 f"{int(Hh['관측연수'].sum())}해 (설계서 10절 3번: 87개 1,096해)")
    NOTES.append(f"하루 창: A의 정의(날짜숫자 고정)로 다시 센 값이 A의 표와 같은 조합 {n_a}/{n}, "
                 f"함수(달과 날 고정)가 A의 하루_적중률과 같은 조합 {n_md}/{n}. 윤년에는 같은 달과 날이 하루 밀려 몇 조합은 다를 수 있다")
    if same:
        check(n == len(Hh) and n_a == n, "10절: A의 정의로 다시 센 하루_적중률이 A의 표와 다름")
        check(set(zip(F._H["수목원"], F._H["종"])) == set(zip(Hh["수목원"], Hh["종"])), "10절: 8년 이상 조합이 A와 다름")
    else:
        NOTES.append("정제본 판의 조합 통계가 A의 표와 달라 10절 맞춰 보기는 참고로만 본다(점검용 자료이거나 판이 다르다).")


def main():
    print("판:", F.SOURCE, "| 서비스 표:", "있음" if F.FINAL else "없음", "| 가정 입력 기울기:", F.SLOPE,
          "| 최근 5년 보정 채택:", F.USE_M0R, "| M1 채택:", F.USE_M1)
    site, sp, au = "국립수목원", "진달래", "단풍나무"
    t_lists()
    t_predict()
    t_scenario(site, sp, au)
    t_lead(site, sp)
    t_predict_errors(site, sp, au)
    t_overlap()
    t_overlap_errors(site, sp, au)
    t_recommend()
    t_recommend_errors(site)
    t_nfc(site, sp)
    t_internal(site, sp)
    t_against_a()
    for n in NOTES:
        print("참고:", n)
    print(f"통과 {N_OK[0]}개, 실패 {len(FAILS)}개")
    if FAILS:
        for f in FAILS[:60]:
            print("!!", f)
        if len(FAILS) > 60:
            print(f"!! 그 밖에 {len(FAILS) - 60}개")
        sys.exit(1)


if __name__ == "__main__":
    main()
