# 기능 함수 점검 반환 키와 범위와 오류 코드 확인 후 실패 시 종료 코드 1

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
    "forecast": ["ok", "site", "species", "phenomenon", "stage", "year", "lead", "method", "pred_doy", "pred_date",
                     "interval", "mean_doy", "diff_from_mean_days", "sd_days", "n_years", "actual_doy", "history",
                     "scenario", "warnings", "source"],
    "overlap": ["ok", "site", "species", "phenomenon", "lead", "start_date", "days", "end_date",
                         "min_overlap_days", "bloom_length_days", "bloom_length_source", "prob", "headline", "grade",
                         "expected_overlap_days", "best", "by_days", "by_start", "past", "scenario", "basis",
                         "n_years", "warnings", "source"],
    "ranking": ["ok", "site", "phenomenon", "start_date", "end_date", "min_overlap_days", "n_candidates",
                          "fit_method", "ranking", "excluded", "warnings", "source"],
    "revision": ["ok", "site", "species", "phenomenon", "stage", "year", "ref", "ref_doy", "ref_date", "state", "base",
                 "revised", "actual_doy", "actual_date", "years", "warnings", "source"],
}
INTERVAL = ["level", "low_doy", "high_doy", "low_date", "high_date"]
WINDOW = ["start_doy", "start_date", "end_doy", "end_date"]
STEP = ["method", "pred_doy", "pred_date", "interval", "error_days"]
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

def fact(cond, what):
    if cond:
        N_OK[0] += 1
    else:
        FAILS.append(what)
    return bool(cond)

# 함수 호출
# 예외가 나면 실패로 기록하고 None 반환
def trial(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        FAILS.append(f"예외 {fn.__name__}{a}: {type(e).__name__}: {e}")
        return None

def fields(d, keys, what):
    miss = [k for k in keys if not isinstance(d, dict) or k not in d]
    return fact(not miss, f"{what}: 빠진 키 {miss}")

def cols(df, want, what, exact=False):
    ok = isinstance(df, pd.DataFrame) and (list(df.columns) == want if exact else all(c in df.columns for c in want))
    return fact(ok, f"{what}: 열이 약속과 다름 {list(df.columns) if isinstance(df, pd.DataFrame) else type(df)}")

def code(r, want, what):
    got = r.get("code") if isinstance(r, dict) else None
    ok = isinstance(r, dict) and r.get("ok") is False and got == want and isinstance(r.get("message"), str) \
        and "source" in r and isinstance(r.get("warnings"), list)
    return fact(ok, f"{what}: {want}를 기대했는데 {got}")

def iso(year, k):
    return dt.date(year, 1, 1) + dt.timedelta(days=int(k) - 1)

# 지난 해 세기를 함수와 따로 다시 계산
# 축제는 달과 날로 맞추고 날마다 꽃이 피어 있는지 확인
def tally(site, sp, start, days, m=1, length=None):
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

def listing():
    S = trial(F.sites)
    cols(S, SITE_COLS, "sites", exact=True)
    fact(S is not None and len(S) == 10, "sites: 10곳이 아님")
    for site in F.SITES:
        for phen, pool in (("개화", F.SPRING_SPECIES), ("단풍", F.AUTUMN_SPECIES)):
            L = trial(F.catalog, site, phen)
            if cols(L, SPECIES_COLS, f"catalog {site} {phen}", exact=True):
                fact(L["종"].tolist() == pool, f"catalog {site} {phen}: 종 목록이 다름")
                fact(phen == "개화" or not L["추천가능"].any(), f"catalog {site} 단풍: 추천가능이 참")
    code(trial(F.catalog, "없는곳"), "UNKNOWN_SITE", "catalog 없는 곳")
    code(trial(F.catalog, F.SITES[0], "낙엽"), "UNKNOWN_PHENOMENON", "catalog 없는 현상")

def forecasts():
    for site in F.SITES:
        for phen, pool in (("개화", F.SPRING_SPECIES), ("단풍", F.AUTUMN_SPECIES)):
            L = F.catalog(site, phen).set_index("종")
            for sp in pool:
                what = f"forecast {site} {sp}"
                r = trial(F.forecast, site, sp, phen)
                if r is None:
                    continue
                fact(r.get("ok") == bool(L.loc[sp, "예측가능"]), f"{what}: ok와 catalog의 예측가능이 다름")
                if not r.get("ok"):
                    fact(r.get("code") in ("NO_DATA", "TOO_FEW_YEARS") and "n_years" in r,
                          f"{what}: 실패 코드 {r.get('code')}")
                    continue
                fields(r, KEYS["forecast"], what)
                fields(r["interval"], INTERVAL, what + " interval")
                iv = r["interval"]
                fact(iv["low_doy"] <= r["pred_doy"] <= iv["high_doy"]
                      and iv["low_date"] <= r["pred_date"] <= iv["high_date"], f"{what}: 범위 순서")
                fact(r["n_years"] >= F.MIN_YEARS and r["source"] == F.SOURCE and isinstance(r["warnings"], list),
                      f"{what}: n_years, source, warnings")
                fact(r["scenario"] is None and r["actual_doy"] is None and r["lead"] == "계획",
                      f"{what}: 2027년 계획인데 scenario나 actual_doy가 있음")
                want = round(F.RECENT5[phen], 1) if F.USE_M0R.get(phen) else 0.0
                fact(abs(r["diff_from_mean_days"] - want) < 0.051, f"{what}: 최근 5년 보정값이 판정과 다름")
                widths = [F.forecast(site, sp, phen, level=lv)["interval"] for lv in (0.5, 0.8, 0.9)]
                widths = [q["high_doy"] - q["low_doy"] for q in widths]
                fact(widths[0] < widths[1] < widths[2], f"{what}: 범위 수준이 높을수록 넓어야 함 {widths}")
                a, b = F.combo(site, sp, "개화시작"), F.combo(site, sp, "낙화90~100%")
                want_bw = phen == "개화" and a is not None and b is not None \
                    and min(a["관측연수"], b["관측연수"]) >= F.MIN_YEARS
                fact(("bloom_window" in r) == want_bw, f"{what}: bloom_window 유무가 규칙과 다름")
                if "bloom_window" in r:
                    bw = r["bloom_window"]
                    fields(bw, WINDOW, what + " bloom_window")
                    fact(bw["start_doy"] <= bw["end_doy"], f"{what}: bloom_window 순서")
                h = r["history"]
                if not F.FINAL:
                    fact(h is None, f"{what}: 집계표 판인데 history가 있음")
                    continue
                if cols(h, HISTORY, what + " history", exact=True):
                    fact(len(h) == int(F.combo(site, sp, r["stage"])["관측연수"]), f"{what}: history 줄 수")
                    y0, v0 = int(h["연도"].iloc[-1]), float(h["날짜숫자"].iloc[-1])
                    q = F.forecast(site, sp, phen, year=y0)
                    fact(q["actual_doy"] == v0 and any("지난 해를 골랐습니다" in w for w in q["warnings"]),
                          f"{what}: 지난 해 {y0}의 actual_doy나 경고")

def scenarios(site, sp, au):
    base = F.forecast(site, sp)
    bw = base["interval"]["high_doy"] - base["interval"]["low_doy"]
    for x in (-2, -1.0, 0.5, 2.0):
        r = F.forecast(site, sp, temp_scenario=x)
        what = f"forecast 기온 가정 {x}"
        if fields(r.get("scenario"), SCENARIO, what):
            fact(r["scenario"]["shift_days"] == round(F.SLOPE * x, 1), f"{what}: shift_days")
        fact(abs((r["pred_doy"] - base["pred_doy"]) - F.SLOPE * x) <= 0.11, f"{what}: 예상 날짜가 기울기만큼 옮겨지지 않음")
        fact(abs((r["interval"]["high_doy"] - r["interval"]["low_doy"]) - bw) <= 0.11, f"{what}: 범위 폭이 바뀜")
        fact(r["lead"] == "시나리오", f"{what}: lead")
    r = F.forecast(site, au, "단풍", temp_scenario=1.0)
    fact(r.get("ok") is not True or (r["scenario"] is None and any("단풍에는 기온 가정" in w for w in r["warnings"])),
          "forecast 단풍 기온 가정: 반영하지 않는다는 경고가 없음")
    f = F.overlap(site, sp, "2027-04-01", 7, temp_scenario=1.0)
    if fields(f.get("scenario"), SCENARIO, "overlap 기온 가정"):
        fact(f["lead"] == "시나리오" and f["headline"]["basis"] == "분포" and f["basis"].endswith("+ 기온 가정"),
              "overlap 기온 가정: lead, headline, basis")

def leads(site, sp):
    r = F.forecast(site, sp, lead="2월 말")
    if F.FINAL and F.USE_M1:
        fact(r["lead"] == "계획" and any("2월 기온이 아직 없어" in w for w in r["warnings"]),
              "forecast 2월 말(입력 없는 해): 계획 값과 경고")
        y = int(F._M1X["연도"].max())
        q = F.forecast(site, sp, year=y, lead="2월 말")
        fact(q["method"] == "2월 말 예측(M1)" and q["lead"] == "2월 말", f"forecast 2월 말({y}): M1을 쓰지 않음")
        lo, hi = q["interval"]["low_doy"], q["interval"]["high_doy"]
        fact(lo <= q["pred_doy"] <= hi, f"forecast 2월 말({y}): 범위 순서")
    else:
        fact(r["lead"] == "계획" and r["method"] in ("기준(과거 평균)", "최근 5년 보정") and r["warnings"],
              "forecast 2월 말: M1을 안 쓰는데 계획 값이 아니거나 경고가 없음")
    q = F.forecast(site, sp, lead="엉뚱한값")
    fact(q.get("ok") and any("lead는" in w for w in q["warnings"]), "forecast lead 오입력: 경고가 없음")
    # 시나리오는 기온 가정이 있어야 사용 없으면 계획 값과 경고 반환
    need = "temp_scenario"
    q = F.forecast(site, sp, lead="시나리오")
    fact(q.get("ok") and q["lead"] == "계획" and any(need in w for w in q["warnings"]),
          "forecast 기온 가정 없는 시나리오: 계획 값과 경고")
    q = F.forecast(site, sp, lead="시나리오", temp_scenario=1.0)
    fact(q.get("ok") and q["lead"] == "시나리오" and not any(need in w or "lead는" in w for w in q["warnings"]),
          "forecast 기온 가정 있는 시나리오: lead나 경고")
    q = F.forecast(site, sp)
    fact(q.get("ok") and not any(need in w or "lead는" in w for w in q["warnings"]), "forecast 계획: lead 경고가 붙음")
    FO = F.overlap
    for ld, want in (("시나리오", need), ("2월 말", "2월 말 예측을 쓰지 않아"), ("엉뚱한값", "lead는")):
        f = FO(site, sp, "2027-04-01", 7, lead=ld)
        fact(f.get("ok") and f["lead"] == "계획" and any(want in w for w in f["warnings"]),
              f"overlap lead={ld}: 계획 값과 경고")
    for ld, x, want in (("시나리오", 1.0, "시나리오"), ("계획", None, "계획")):
        f = FO(site, sp, "2027-04-01", 7, lead=ld, temp_scenario=x)
        fact(f.get("ok") and f["lead"] == want and not any(need in w or "lead는" in w for w in f["warnings"]),
              f"overlap lead={ld}: lead나 경고")

def forecasterrors(site, sp, au):
    P = F.forecast
    for y in (2008, 2031, "2027", True, 2027.0, None):
        code(trial(P, site, sp, year=y), "BAD_YEAR", f"forecast year={y!r}")
    for lv in (0.7, "0.8", True, None, [0.8]):
        code(trial(P, site, sp, level=lv), "BAD_LEVEL", f"forecast level={lv!r}")
    for x in (3, -2.5, float("nan"), float("inf"), "1", True, [1]):
        code(trial(P, site, sp, temp_scenario=x), "BAD_SCENARIO", f"forecast temp_scenario={x!r}")
    code(trial(P, "없는곳", sp), "UNKNOWN_SITE", "forecast 없는 곳")
    code(trial(P, ["국립수목원"], sp), "UNKNOWN_SITE", "forecast 목록으로 넣은 곳")
    code(trial(P, site, "장미"), "UNKNOWN_SPECIES", "forecast 없는 종")
    code(trial(P, site, au), "UNKNOWN_SPECIES", "forecast 개화에 단풍 종")
    code(trial(P, site, sp, "낙엽"), "UNKNOWN_PHENOMENON", "forecast 없는 현상")

def overlaps():
    for site in F.SITES:
        for sp in F.SPRING_SPECIES:
            what = f"overlap {site} {sp}"
            o = F.combo(site, sp, "개화시작")
            start = iso(2027, round(float(o["평균"]))) if o is not None else dt.date(2027, 4, 1)
            r = trial(F.overlap, site, sp, start.isoformat(), 7)
            if r is None:
                continue
            if not r.get("ok"):
                fact(r.get("code") in ("NO_DATA", "TOO_FEW_YEARS"), f"{what}: 실패 코드 {r.get('code')}")
                continue
            fields(r, KEYS["overlap"], what)
            fields(r["headline"], HEADLINE, what + " headline")
            fields(r["best"], BEST, what + " best")
            fact(0 <= r["prob"] <= 1 and 0 <= r["best"]["prob"] <= 1, f"{what}: 확률이 0~1 밖")
            fact(r["best"]["prob"] >= r["prob"], f"{what}: 가장 좋은 기간의 확률이 입력한 기간보다 낮음")
            fact(r["end_date"] == (start + dt.timedelta(days=6)).isoformat(), f"{what}: end_date")
            bd, bs = r["by_days"], r["by_start"]
            if cols(bd, BY_DAYS, what + " by_days", exact=True):
                fact(len(bd) == 30 and bd["확률"].is_monotonic_increasing and bd["기대겹침일수"].is_monotonic_increasing
                      and (bd["최적기간_확률"] >= bd["확률"]).all(), f"{what}: by_days 곡선 모양")
                fact(abs(bd.loc[bd["일수"] == 7, "확률"].iloc[0] - r["prob"]) < 1e-9, f"{what}: by_days와 prob가 다름")
            if cols(bs, BY_START, what + " by_start", exact=True):
                fact(bs["확률"].between(0, 1).all(), f"{what}: by_start 확률 범위")
            fact(r["grade"] == F.grade(r["headline"]["value"]), f"{what}: grade가 headline과 다름")
            fact(0 <= r["expected_overlap_days"] <= r["days"], f"{what}: 기대 겹침 일수 범위")
            m3 = F.overlap(site, sp, start, 7, min_overlap_days=3)
            fact(m3["prob"] <= r["prob"], f"{what}: 겹침 기준을 3일로 올렸는데 확률이 올라감")
            fact(F.overlap(site, sp, start, 7)["prob"] == r["prob"], f"{what}: date와 글자 입력 결과가 다름")
            p = r["past"]
            if not F.FINAL:
                fact(p is None and r["headline"]["basis"] == "분포", f"{what}: 집계표 판인데 past가 있음")
                continue
            if not fields(p, PAST, what + " past"):
                continue
            yrs = sorted(F._B.loc[(F._B["수목원"] == site) & (F._B["종"] == sp), "연도"].astype(int))
            fact(sorted(p["years_overlap"] + p["years_miss"]) == yrs and p["n_years"] == len(yrs)
                  and p["n_overlap"] == len(p["years_overlap"]), f"{what}: past의 해 목록")
            fact(p["n_overlap"] == tally(site, sp, start, 7), f"{what}: 따로 센 겹친 해 수와 다름")
            if p["n_years"] >= F.MIN_YEARS:
                fact(r["headline"]["basis"] == "과거 해" and r["headline"]["value"] == p["rate"]
                      and r["headline"]["text"] == f"{p['n_years']}해 중 {p['n_overlap']}해 겹침",
                      f"{what}: headline이 past와 다름")
            L = F.overlap(site, sp, start, 7, bloom_length=10)
            fact(L["bloom_length_source"] == "직접 입력" and L["bloom_length_days"] == 10
                  and L["past"]["n_overlap"] == tally(site, sp, start, 7, length=10),
                  f"{what}: 꽃 시기 길이를 넣었을 때 지난 해 세기")

def overlaperrors(site, sp, au):
    FO = F.overlap
    for d in ("2027-13-01", "2027-02-30", "abc", None, ""):
        code(trial(FO, site, sp, d, 7), "BAD_DATE", f"overlap start_date={d!r}")
    for n in (0, 61, 2.5, True, "7", None):
        code(trial(FO, site, sp, "2027-04-01", n), "BAD_DAYS", f"overlap days={n!r}")
    for n in (0, 61, 2.5):
        code(trial(FO, site, sp, "2027-04-01", 7, max_days=n), "BAD_DAYS", f"overlap max_days={n!r}")
    for n in (0, 8, 1.5, True):
        code(trial(FO, site, sp, "2027-04-01", 7, min_overlap_days=n), "BAD_DAYS",
             f"overlap min_overlap_days={n!r}")
    for L in (0, 61, True, "10", float("nan")):
        code(trial(FO, site, sp, "2027-04-01", 7, bloom_length=L), "BAD_LENGTH", f"overlap bloom_length={L!r}")
    code(trial(FO, site, sp, "2027-04-01", 7, temp_scenario=3), "BAD_SCENARIO", "overlap temp_scenario=3")
    code(trial(FO, site, au, "2027-10-20", 7, "단풍"), "NOT_SUPPORTED", "overlap 단풍")
    code(trial(FO, "없는곳", sp, "2027-04-01", 7), "UNKNOWN_SITE", "overlap 없는 곳")
    code(trial(FO, site, "장미", "2027-04-01", 7), "UNKNOWN_SPECIES", "overlap 없는 종")
    r = trial(FO, site, sp, "2028-02-29", 7)
    fact(r is not None and r.get("ok"), "overlap 2월 29일 입력이 실패함")
    # 축제 시작일의 해도 기능 1과 같은 범위만 허용
    for d in ("2008-12-31", "2031-01-01", "9999-12-31", "0001-01-01", dt.date(9999, 12, 31)):
        code(trial(FO, site, sp, d, 7), "BAD_DATE", f"overlap 범위 밖 해 {d!r}")
    for d in ("2009-01-01", "2030-12-31"):
        r = trial(FO, site, sp, d, 7)
        fact(r is not None and r.get("ok"), f"overlap 범위 끝 해 {d}가 실패함")

def rankings():
    for site in F.SITES:
        for a, b in WINDOWS:
            what = f"ranking {site} {a}~{b}"
            r = trial(F.ranking, site, a, b)
            if r is None:
                continue
            if not r.get("ok"):
                code(r, "NO_CANDIDATE", what)
                fact(isinstance(r.get("excluded"), pd.DataFrame), f"{what}: excluded가 없음")
                continue
            fields(r, KEYS["ranking"], what)
            R, X = r["ranking"], r["excluded"]
            if not (cols(R, RANKING, what + " ranking") and cols(X, EXCLUDED, what + " excluded", exact=True)):
                continue
            fact(R["순위"].tolist() == list(range(1, len(R) + 1)) and r["n_candidates"] == len(R), f"{what}: 순위 번호")
            key = list(zip(R["적합도"], R["안정성점수"].fillna(-1)))
            fact(all(key[i] >= key[i + 1] for i in range(len(key) - 1)), f"{what}: 적합도와 안정성점수 순서")
            fact(set(R["종"]).isdisjoint(X["종"]) and set(R["종"]) | set(X["종"]) == set(F.SPRING_SPECIES),
                  f"{what}: 순위와 제외 목록이 11종을 나누지 않음")
            want = {sp for sp in F.SPRING_SPECIES if F.hitrate(site, sp) is not None and "ok" not in F.onset(site, sp)}
            fact(set(R["종"]) == want, f"{what}: 후보가 개화시작과 낙화 8년 이상 조합과 다름")
            fact(R["적합도"].between(0, 1).all() and (R["등급"] == R["적합도"].map(F.grade)).all(), f"{what}: 적합도와 등급")
            fact(R["적합도"].max() > 0, f"{what}: 적합도가 모두 0인데 ok")
            fact(r["fit_method"] == ("지난 해 세기" if F.FINAL else "정규 근사(임시)"), f"{what}: fit_method")
            days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1
            for _, row in R.iterrows():
                fo = F.overlap(site, row["종"], a, days)
                if F.FINAL:
                    p = fo["past"]
                    fact(row["겹친해수"] == p["n_overlap"] and row["관측연수"] == p["n_years"]
                          and abs(row["적합도"] - p["n_overlap"] / p["n_years"]) <= 0.005 + 1e-9,
                          f"{what} {row['종']}: 기능 3의 적합도가 기능 2의 지난 해 세기와 다름")
                else:
                    fact(abs(row["적합도"] - fo["prob"]) <= 0.0051 and row["겹친해수"] is None,
                          f"{what} {row['종']}: 기능 3의 적합도가 기능 2의 확률과 다름")
    one = trial(F.ranking, F.SITES[0], "2027-04-10", "2027-04-10")
    fact(one is not None and one.get("ok") is not None, "ranking 하루 창")

def rankingerrors(site):
    RS = F.ranking
    code(trial(RS, site, "2027-04-10", "2027-04-01"), "BAD_DATE", "ranking 종료일이 시작일보다 앞")
    code(trial(RS, site, "2027-12-25", "2028-01-05"), "BAD_DATE", "ranking 해가 다름")
    code(trial(RS, site, "2027-02-30", "2027-03-05"), "BAD_DATE", "ranking 없는 날짜")
    code(trial(RS, site, "2027-04-01", "2027-04-07", min_overlap_days=0), "BAD_DAYS", "ranking 겹침 0일")
    code(trial(RS, site, "2027-04-01", "2027-04-07", min_overlap_days=8), "BAD_DAYS", "ranking 겹침 8일")
    code(trial(RS, site, "2027-10-01", "2027-10-07", "단풍"), "NOT_SUPPORTED", "ranking 단풍")
    code(trial(RS, "없는곳", "2027-04-01", "2027-04-07"), "UNKNOWN_SITE", "ranking 없는 곳")
    for a, b in (("2008-04-01", "2008-04-07"), ("2031-04-01", "2031-04-07"), ("9999-12-25", "9999-12-31")):
        code(trial(RS, site, a, b), "BAD_DATE", f"ranking 범위 밖 해 {a}")
    # 꽃 철이 아닌 기간은 적합도 0인 순위표 대신 후보 없음으로 응답
    for a, b in (("2027-12-01", "2027-12-07"), ("2027-01-10", "2027-01-20")):
        r = trial(RS, site, a, b)
        if code(r, "NO_CANDIDATE", f"ranking 꽃 철이 아닌 기간 {a}"):
            cols(r.get("excluded"), EXCLUDED, f"ranking 꽃 철이 아닌 기간 {a} excluded", exact=True)
    saved = F._H
    F._H = F._H.iloc[0:0]
    try:
        r = trial(RS, site, "2027-04-01", "2027-04-07")
    finally:
        F._H = saved
    if code(r, "NO_CANDIDATE", "ranking 후보 없음"):
        fact(isinstance(r.get("excluded"), pd.DataFrame) and len(r["excluded"]) == len(F.SPRING_SPECIES),
              "ranking 후보 없음: excluded")

# 기능 4 점검
# 표의 줄마다 반환 키와 범위 순서와 표의 값과 기능 1의 실제 날짜 확인
# 표가 없는 판에서는 오류 코드만 확인
def revisions(site, sp, au):
    P = F.revision
    code(trial(P, "없는곳", sp, 2020), "UNKNOWN_SITE", "revision 없는 곳")
    code(trial(P, site, "장미", 2020), "UNKNOWN_SPECIES", "revision 없는 종")
    code(trial(P, site, au, 2020), "UNKNOWN_SPECIES", "revision 단풍 종")
    for y in (2008, 2031, "2020", True, 2020.0, None):
        code(trial(P, site, sp, y), "BAD_YEAR", f"revision year={y!r}")
    path = F.SERVICE / "revision.csv"
    T = F.table(path) if F.FINAL and path.exists() else None
    if T is None or T.empty:
        code(trial(P, site, sp, 2020), "NOT_SUPPORTED", "revision 표 없음")
        NOTES.append("기준일 예측 표(revision.csv)가 없거나 비어 있어 기능 4는 오류 코드만 본다.")
        return
    fact(not T.duplicated(["수목원", "종", "연도", "기준일"]).any(), "revision 표: 같은 줄이 두 번")
    gain = []
    for (s, k), g in T.groupby(["수목원", "종"], sort=False):
        years = sorted(int(y) for y in g["연도"].unique())
        for y in years:
            what = f"revision {s} {k} {y}"
            t = g[g["연도"] == y].sort_values("기준일_날짜숫자").iloc[-1]
            r = trial(P, s, k, y)
            if not (fact(isinstance(r, dict) and r.get("ok") is True, f"{what}: ok가 아님") and fields(r, KEYS["revision"], what)):
                continue
            rd = r["ref_doy"]
            fact(r["ref"] == t["기준일"] and rd == int(t["기준일_날짜숫자"]) and r["ref_date"] == iso(y, rd).isoformat()
                  and r["state"] == t["상태"] and r["years"] == years and r["source"] == F.SOURCE
                  and isinstance(r["warnings"], list), f"{what}: 기준일이나 상태나 해 목록이 표와 다름")
            fact(r["actual_doy"] == F.forecast(s, k, year=y)["actual_doy"] == float(t["날짜숫자"]) and r["actual_doy"] > rd,
                  f"{what}: 실제 날짜가 기능 1과 다르거나 기준일보다 이름")
            for key, side in (("base", "기준"), ("revised", "고친")):
                q = r[key]
                if not (fields(q, STEP, f"{what} {key}") and fields(q["interval"], INTERVAL, f"{what} {key} interval")):
                    continue
                iv, want = q["interval"], float(t[f"{side}_예측"])
                fact(rd < iv["low_doy"] <= q["pred_doy"] <= iv["high_doy"] and iv["level"] == 0.8
                      and iv["low_date"] <= q["pred_date"] <= iv["high_date"], f"{what} {key}: 범위 순서나 기준일보다 이른 범위")
                fact(q["pred_doy"] == round(want, 1) and q["error_days"] == round(abs(want - float(t["날짜숫자"])), 1),
                      f"{what} {key}: 표의 값과 다름")
            gain.append(r["base"]["error_days"] - r["revised"]["error_days"])
    fact(len(gain) == len(T.drop_duplicates(["수목원", "종", "연도"])) and sum(gain) > 0,
          "revision: 채택된 모델인데 평균 오차가 기준 모델보다 작지 않음")
    s, k = T["수목원"].iloc[0], T["종"].iloc[0]
    years = sorted(int(y) for y in T.loc[(T["수목원"] == s) & (T["종"] == k), "연도"].unique())
    r = trial(P, s, k, 2027)
    code(r, "NO_DATA", "revision 예측이 없는 해")
    fact(isinstance(r, dict) and r.get("years") == years, "revision 예측이 없는 해: 해 목록")
    a, b = (unicodedata.normalize("NFD", x) for x in (s, k))
    r = trial(P, a, b, years[-1])
    fact(isinstance(r, dict) and r.get("ok") and r["site"] == s and r["species"] == k, "revision 풀린 한글")
    # 그해 기준일에 이미 핀 조합은 이미 피었다는 안내와 함께 자료 없음 코드 반환
    ref = T.groupby("연도")["기준일_날짜숫자"].max()
    H = F._Y[(F._Y["현상"] == F.MAIN_STAGE["개화"]) & F._Y["연도"].isin(ref.index)]
    H = H[H["날짜숫자"] <= H["연도"].map(ref)]
    had = set(zip(T["수목원"], T["종"], T["연도"]))
    H = H[[x not in had for x in zip(H["수목원"], H["종"], H["연도"])]]
    if len(H):
        h = H.iloc[0]
        r = trial(P, h["수목원"], h["종"], int(h["연도"]))
        code(r, "NO_DATA", "revision 이미 핀 해")
        fact(isinstance(r, dict) and "이미 피어" in r.get("message", ""), "revision 이미 핀 해: 안내 글")

# 맥에서 복사한 한글(자모가 풀린 꼴)도 같은 이름으로 받는지 확인
def unicode(site, sp):
    a, b, k, ld = (unicodedata.normalize("NFD", x) for x in (site, sp, "개화", "시나리오"))
    fact(a != site and k != "개화", "풀린 한글 점검: 바꾼 글자가 원래와 같음")
    L = trial(F.catalog, a, k)
    fact(isinstance(L, pd.DataFrame) and L["종"].tolist() == F.SPRING_SPECIES, "catalog 풀린 한글")
    r = trial(F.forecast, a, b, k, lead=ld, temp_scenario=1.0)
    fact(isinstance(r, dict) and r.get("ok") and r["site"] == site and r["species"] == sp
          and not any("lead는" in w for w in r["warnings"]), "forecast 풀린 한글")
    f = trial(F.overlap, a, b, "2027-04-01", 7, k)
    fact(isinstance(f, dict) and f.get("ok") and f["site"] == site and f["species"] == sp
          and f["prob"] == F.overlap(site, sp, "2027-04-01", 7)["prob"], "overlap 풀린 한글")
    g = trial(F.ranking, a, "2027-04-01", "2027-04-07", k)
    fact(isinstance(g, dict) and g.get("ok") and g["site"] == site, "ranking 풀린 한글")

# 함수 안에서 예기치 못한 오류가 나도 예외 대신 내부 오류 코드를 반환하는지 확인
def crash(site, sp):
    def boom(*a, **k):
        raise RuntimeError("점검용 오류")
    for name, fn, args in (("combo", F.forecast, (site, sp)),
                           ("onset", F.overlap, (site, sp, "2027-04-01", 7)),
                           ("onset", F.ranking, (site, "2027-04-01", "2027-04-07")),
                           ("issue", F.revision, (site, sp, 2020))):
        saved = getattr(F, name)
        setattr(F, name, boom)
        try:
            r = trial(fn, *args)
        finally:
            setattr(F, name, saved)
        code(r, "INTERNAL_ERROR", f"{fn.__name__} 내부 오류")

# 10절 맞춰 보기
# 정제본 판에서만 실행
def parity():
    if not F.FINAL:
        NOTES.append("집계표 판이라 10절 맞춰 보기(지난 해 세기와 A의 하루_적중률)는 건너뛴다.")
        return
    A = pd.read_csv(F.TABLES / "variability_site_species.csv")
    J = F._V.merge(A, on=["수목원", "종", "현상"], suffixes=("", "_A"))
    # 관측연수가 1이면 표준편차가 둘 다 빈칸이라 빈칸끼리는 같은 것으로 처리
    sd_same = ((J["표준편차"] - J["표준편차_A"]).abs() < 0.01) | (J["표준편차"].isna() & J["표준편차_A"].isna())
    same = len(J) > 0 and (J["관측연수"] == J["관측연수_A"]).all() \
        and ((J["평균"] - J["평균_A"]).abs() < 0.01).all() and sd_same.all()
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
        r = F.overlap(h["수목원"], h["종"], iso(2027, h["최적고정일"]), 1)
        p = r.get("past") if r.get("ok") else None
        n_md += bool(p) and abs(p["rate"] - h["하루_적중률"]) < 0.0006
    NOTES.append(f"8년 이상 조합: 함수 {len(F._H)}개 {int(F._H['관측연수'].sum())}해, A의 표 {len(Hh)}개 "
                 f"{int(Hh['관측연수'].sum())}해 (설계서 10절 3번: 87개 1,096해)")
    NOTES.append(f"하루 창: A의 정의(날짜숫자 고정)로 다시 센 값이 A의 표와 같은 조합 {n_a}/{n}, "
                 f"함수(달과 날 고정)가 A의 하루_적중률과 같은 조합 {n_md}/{n}. 윤년에는 같은 달과 날이 하루 밀려 몇 조합은 다를 수 있다")
    if same:
        fact(n == len(Hh) and n_a == n, "10절: A의 정의로 다시 센 하루_적중률이 A의 표와 다름")
        fact(set(zip(F._H["수목원"], F._H["종"])) == set(zip(Hh["수목원"], Hh["종"])), "10절: 8년 이상 조합이 A와 다름")
    else:
        NOTES.append("정제본 판의 조합 통계가 A의 표와 달라 10절 맞춰 보기는 참고로만 본다(점검용 자료이거나 판이 다르다).")

def main():
    print("판:", F.SOURCE, "| 서비스 표:", "있음" if F.FINAL else "없음", "| 가정 입력 기울기:", F.SLOPE,
          "| 최근 5년 보정 채택:", F.USE_M0R, "| M1 채택:", F.USE_M1)
    site, sp, au = "국립수목원", "진달래", "단풍나무"
    listing()
    forecasts()
    scenarios(site, sp, au)
    leads(site, sp)
    forecasterrors(site, sp, au)
    overlaps()
    overlaperrors(site, sp, au)
    rankings()
    rankingerrors(site)
    revisions(site, sp, au)
    unicode(site, sp)
    crash(site, sp)
    parity()
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
