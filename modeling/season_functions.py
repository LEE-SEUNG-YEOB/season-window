# 기능 1~3 함수다
# 화면(service 폴더의 앱 파일)이 불러 쓴다
#
# 두 판이 있고 함수 이름과 인자와 반환 키는 같다
#
# 정제본 판은 data/processed/service 폴더에 서비스 표가 있으면 그것을 읽는다
# 서비스 표는 modeling 폴더의 6단계 파일이 만든다
# 이 판의 source는 정제본이다
# 기능 2의 대표값과 기능 3의 적합도는 지난 해마다 실제로 겹쳤는지 센 값이다
# prob와 곡선 두 개(일수별과 시작일별)와 가장 좋은 기간(best)은 근사다
# 개화시작 분포와 고정한 꽃 시기 길이로 낸다
#
# 집계표 판은 서비스 표가 없을 때 reports/tables의 집계표로 근사한다
# 이 판의 source는 집계표(임시)다
# 개화시작이 정규분포(조합 평균과 조합 표준편차)를 따른다고 본 근사다
# 그래서 팀이 정한 겹친 해의 비율과 정의가 같지 않다
# 이 판의 숫자는 화면 개발용이고 제출물과 발표 자료에 옮기지 않는다
#
# 기능 1~3은 언제나 dict(이름과 값의 묶음)를 돌려주고 예외를 던지지 않는다
# 실패하면 ok가 False이고 code와 한글 message가 있다
# 실패해도 source와 warnings(빈 목록) 키는 늘 있다
# 꽃이 핀 날은 개화시작 <= 날 <= 낙화90~100%로 센다(A의 집계와 같은 정의)
# 저장소 어느 폴더에 두어도 돌아간다(위로 올라가며 reports/tables를 찾는다)
# 점검할 때는 환경변수로 저장소 위치를 바꿀 수 있다

import datetime as dt
import functools
import math
import numbers
import os
import unicodedata
from pathlib import Path

import pandas as pd


# 저장소 맨 위 폴더를 찾는다
# 저장소 위치 환경변수가 있으면 그곳을 쓴다
# 없으면 이 파일에서 위로 올라가며 reports/tables가 있는 곳을 쓴다
def _find_root():
    env = os.environ.get("SEASON_ROOT")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "reports" / "tables").is_dir() and (p / "reference").is_dir():
            return p
    raise FileNotFoundError("reports/tables 폴더를 찾지 못했습니다. 저장소 안에서 불러 주세요.")


ROOT = _find_root()
TABLES, REFERENCE = ROOT / "reports" / "tables", ROOT / "reference"
SERVICE = ROOT / "data" / "processed" / "service"
SERVICE_FILES = ("combo_stats.csv", "history.csv", "bloom_pairs.csv", "gaps.csv", "shift.csv", "decision.csv")
FINAL = all((SERVICE / f).exists() for f in SERVICE_FILES)
SOURCE = "정제본" if FINAL else "집계표(임시)"

SPRING_SPECIES = ["왕벚나무", "진달래", "산수유", "미선나무", "백목련", "개나리",
                  "생강나무", "아까시나무", "철쭉", "산철쭉", "마가목"]
AUTUMN_SPECIES = ["단풍나무", "당단풍나무", "은행나무", "신갈나무", "졸참나무"]
MIN_YEARS = 8
MAIN_STAGE = {"개화": "개화50%이상", "단풍": "단풍50%이상"}
# 범위를 잡을 때 쓰는 t 분포 분위수다
# 바깥 열쇠는 범위 수준이고 안쪽 열쇠는 자유도(관측연수 - 1)다
# 관측한 해가 적을수록 정규분포보다 범위를 넓게 잡는다
T_Q = {
    0.5: {7: 0.711, 8: 0.706, 9: 0.703, 10: 0.700, 11: 0.697, 12: 0.695, 13: 0.694, 14: 0.692, 15: 0.691, 16: 0.690},
    0.8: {7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372, 11: 1.363, 12: 1.356, 13: 1.350, 14: 1.345, 15: 1.341, 16: 1.337},
    0.9: {7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746},
}
# 등급 기준이다(임시)
# 첫 값 이상은 높음이다
# 둘째 값 이상은 보통이고 그 아래는 낮음이다
GRADE_CUTS = (0.8, 0.5)
SCENARIO_MAX = 2.0               # 기온 가정을 받는 범위(도)
SCENARIO_WINDOW = "2~3월 평균기온(평년 대비)"
LEADS = ("계획", "2월 말", "시나리오")
YEAR_LO, YEAR_HI = 2009, 2030    # 고를 수 있는 해(첫 관측 해부터 2030년까지)


def _read(p):
    return pd.read_csv(p, encoding="utf-8-sig")


_M = _read(REFERENCE / "station_map.csv")
SITES = _M["수목원"].tolist()

if FINAL:
    _V = _read(SERVICE / "combo_stats.csv")                  # 수목원 x 종 x 단계별 관측연수와 평균과 표준편차
    _G = _read(SERVICE / "gaps.csv")
    _GC = _G[_G["구분"] == "조합"]
    _GS = _G[_G["구분"] == "종"].set_index("종")
    # 개화시작과 낙화가 함께 8년 이상인 조합이다
    # 집계표 판에서 쓰는 A의 고정일 적중 표와 같은 모양이다
    _H = _GC[_GC["기간_해수"] >= MIN_YEARS].rename(columns={"기간_해수": "관측연수", "기간_중앙값": "개화기간_중앙값"})
    # 종 중앙값 꽃 시기 길이다
    _D = _GS["기간_중앙값"]
    _B = _read(SERVICE / "bloom_pairs.csv")                  # 해별 개화시작과 낙화
    _Y = _read(SERVICE / "history.csv")                      # 해별 날짜
    _S = _read(SERVICE / "shift.csv")
    _DEC = _read(SERVICE / "decision.csv")
    RECENT5 = {k: float(_S.loc[_S["현상"] == k, "최근5년_편차"].iloc[0]) for k in MAIN_STAGE if (_S["현상"] == k).any()}
    RECENT_SPAN = {r["현상"]: f"{int(r['시작연도'])}~{int(r['끝연도'])}" for _, r in _S.iterrows()}
    LAST_YEAR = int(_B["연도"].max()) if len(_B) else None

    def _adopted(item, phen):
        r = _DEC[(_DEC["항목"] == item) & (_DEC["현상"] == phen)]
        return bool(len(r)) and r["판정"].iloc[0] == "채택"

    USE_M0R = {"개화": _adopted("M0r", "개화"), "단풍": _adopted("M0r", "단풍")}
    USE_M1 = _adopted("M1", "개화")
    _M1 = _read(SERVICE / "m1.csv") if (SERVICE / "m1.csv").exists() else None
    _M1X = _read(SERVICE / "m1_inputs.csv") if (SERVICE / "m1_inputs.csv").exists() else None
    _sl = _DEC[_DEC["항목"] == "가정 입력 기울기"]
    SLOPE = float(_sl["값"].iloc[0]) if len(_sl) and pd.notna(_sl["값"].iloc[0]) else None
else:
    _V = _read(TABLES / "variability_site_species.csv")      # 수목원 x 종 x 단계별 관측연수와 평균과 표준편차
    _H = _read(TABLES / "eda_fixed_day_hit.csv")             # 개화시작과 낙화가 함께 8년 이상인 조합 87개
    _D = _read(TABLES / "eda_bloom_duration_species.csv").set_index("종")["중앙값"]
    _GC = _GS = _B = _Y = _M1 = _M1X = None
    # 전국 연도별 편차(일)를 읽는다
    _A = _read(TABLES / "eda_year_anomaly.csv").sort_values("연도")
    # 전국 최근 5년 편차(일)를 낸다
    RECENT5 = {k: float(_A[v].tail(5).mean()) for k, v in MAIN_STAGE.items()}
    _yrs = [int(y) for y in _A["연도"].tail(5)]
    RECENT_SPAN = {k: f"{_yrs[0]}~{_yrs[-1]}" for k in MAIN_STAGE}
    LAST_YEAR = _yrs[-1]
    USE_M0R = {"개화": False, "단풍": True}   # 단풍은 설계서 6절 4항대로 최근 5년 보정을 넣는다
    USE_M1 = False
    SLOPE = None
if SLOPE is None:   # 2~3월 평균기온 1도당 일수(전국 공통)는 A의 표 값을 쓴다
    SLOPE = float(_read(TABLES / "eda_temp_pooled.csv")["기울기_일per도"].iloc[0])


def _fail(code, message, **extra):
    return {"ok": False, "code": code, "message": message, "source": SOURCE, "warnings": [], **extra}


# 기능 1~3이 예외를 던지지 않게 감싼다
# 예기치 못한 오류는 내부 오류 코드로 돌려준다
def _safe(fn):
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # 화면이 멈추지 않게 한다
            return _fail("INTERNAL_ERROR", "계산 중 예기치 못한 오류가 났습니다.", detail=f"{type(e).__name__}: {e}")
    return wrap


def _is_int(x):
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


# 맥에서 복사한 한글은 자모가 풀린 꼴로 올 때가 있다
# 이름을 찾기 전에 글자를 하나로 모은 꼴로 바꾼다
def _nfc(x):
    return unicodedata.normalize("NFC", x) if isinstance(x, str) else x


def _cdf(x, mu, sd):
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))


def _tq(level, n):
    return T_Q[level][min(max(n - 1, 7), 16)]


# 그해 날짜숫자를 날짜 글자로 바꾼다
# 날짜숫자는 반올림하고 딱 절반이면 올린다
def _iso(year, doy):
    return (dt.date(year, 1, 1) + dt.timedelta(days=int(math.floor(doy + 0.5)) - 1)).isoformat()


def _parse(d):
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))


# d의 월과 일을 그해의 날짜숫자로 바꾼다
# 2월 29일은 평년이면 2월 28일로 본다
def _doy(year, d):
    try:
        return dt.date(year, d.month, d.day).timetuple().tm_yday
    except ValueError:
        return dt.date(year, 2, 28).timetuple().tm_yday


def _row(site, species, stage):
    r = _V[(_V["수목원"] == site) & (_V["종"] == species) & (_V["현상"] == stage)]
    return None if r.empty else r.iloc[0]


def _hit(site, species):
    r = _H[(_H["수목원"] == site) & (_H["종"] == species)]
    return None if r.empty else r.iloc[0]


def _check(site, phenomenon, species=None):
    if not isinstance(site, str) or site not in SITES:
        return _fail("UNKNOWN_SITE", f"'{site}'은(는) 선택할 수 없는 곳입니다. 가능한 곳: {', '.join(SITES)}")
    if not isinstance(phenomenon, str) or phenomenon not in MAIN_STAGE:
        return _fail("UNKNOWN_PHENOMENON", "현상은 '개화' 또는 '단풍'이어야 합니다.")
    pool = SPRING_SPECIES if phenomenon == "개화" else AUTUMN_SPECIES
    if species is not None and (not isinstance(species, str) or species not in pool):
        return _fail("UNKNOWN_SPECIES", f"'{species}'은(는) {phenomenon} 대상 종이 아닙니다. 가능한 종: {', '.join(pool)}")
    return None


# 기온 가정을 점검한다
# 돌려주는 값은 세 가지다
# 가정 dict 또는 None과 실패 dict 또는 None과 경고 목록이다
def _scenario(temp_scenario, phenomenon):
    if temp_scenario is None:
        return None, None, []
    if isinstance(temp_scenario, bool) or not isinstance(temp_scenario, numbers.Real) \
            or math.isnan(temp_scenario) or abs(temp_scenario) > SCENARIO_MAX:
        return None, _fail("BAD_SCENARIO",
                           f"기온 가정은 -{SCENARIO_MAX:g}~+{SCENARIO_MAX:g}도 사이의 숫자여야 합니다."), []
    if phenomenon != "개화":
        return None, None, ["단풍에는 기온 가정을 반영하지 않습니다."]
    x = float(temp_scenario)
    sc = {"temp_anom": x, "window": SCENARIO_WINDOW, "slope_days_per_degree": SLOPE, "shift_days": round(SLOPE * x, 1)}
    return sc, None, [f"기온 가정은 전국 공통 기울기(1도에 {SLOPE:+.1f}일)를 쓴 근사이고 예측이 아닙니다."]


# 표준편차를 안정성 점수로 바꾼다
# 2일 이하는 100점이고 12일 이상은 0점이다
# 그 사이는 직선으로 잇는다
# 환산식은 설계서에 없어 임시로 정한 것이다
def stability_score(sd):
    if sd is None or pd.isna(sd):
        return None
    return int(round(100 * min(max((12 - float(sd)) / 10, 0), 1)))


# 확률이나 적합도를 세 등급으로 바꾼다
# 기준은 임시이고 팀이 정한다
def grade(value):
    if value is None or pd.isna(value):
        return None
    return "높음" if value >= GRADE_CUTS[0] else ("보통" if value >= GRADE_CUTS[1] else "낮음")


# 화면에 쓸 확률 글자를 만든다
# 5%p 단위로 보여 주고 양 끝은 뭉갠다
def _pct_text(p):
    if p >= 0.95:
        return "95% 이상"
    if p < 0.05:
        return "5% 미만"
    return f"약 {int(round(p * 20)) * 5}%"


# 축제와 꽃 시기가 m일 이상 겹칠 확률을 낸다
# 축제는 s일부터 s+d-1일까지다
# 꽃 시기는 T일부터 T+length일까지다
# T는 개화시작이고 정규분포로 근사한다
def _p_overlap(mu, sd, length, s, d, m=1):
    if d < m:
        return 0.0
    return max(0.0, _cdf(s + d - m + 0.5, mu, sd) - _cdf(s + m - 1 - length - 0.5, mu, sd))


# 축제 d일 가운데 꽃이 피어 있는 날 수의 기댓값을 낸다
def _expected_days(mu, sd, length, s, d):
    return sum(_cdf(k + 0.5, mu, sd) - _cdf(k - length - 0.5, mu, sd) for k in range(s, s + d))


# 꽃 시기 길이(낙화 - 개화시작)와 그 출처를 돌려준다
# 함께 기록된 해가 8년 이상이면 조합 중앙값을 쓴다
# 아니면 종 중앙값을 쓴다
def _length(site, species):
    h = _hit(site, species)
    if h is not None:
        return float(h["개화기간_중앙값"]), f"조합 중앙값({int(h['관측연수'])}년)"
    return (float(_D[species]) if species in _D.index else float("nan")), "종 중앙값"


# 개화시작에서 개화50%이상까지의 간격 중앙값을 돌려준다(정제본 판)
# 8년 미만이면 종 중앙값을 쓴다
# 없으면 None을 돌려준다
def _gap(site, species):
    r = _GC[(_GC["수목원"] == site) & (_GC["종"] == species)]
    if len(r) and r["간격_해수"].iloc[0] >= MIN_YEARS:
        return float(r["간격_중앙값"].iloc[0])
    if species in _GS.index and pd.notna(_GS.loc[species, "간격_중앙값"]):
        return float(_GS.loc[species, "간격_중앙값"])
    return None


# 지난 해마다 축제 기간과 그해 꽃 시기가 겹친 날 수를 센다(정제본 판)
# 축제 날짜는 월과 일을 그해 날짜숫자로 바꿔 맞춘다
# length를 주면 꽃 시기 끝을 개화시작 + length로 본다
# 돌려주는 값은 past dict와 해별 겹친 날 수 dict다
# 기록이 없으면 None 두 개를 돌려준다
def _past(site, species, d0, days, m, length=None):
    if _B is None:
        return None, None
    P = _B[(_B["수목원"] == site) & (_B["종"] == species)].sort_values("연도")
    if P.empty:
        return None, None
    per = {}
    for y, on, off in zip(P["연도"].astype(int), P["개화시작"].astype(float), P["낙화"].astype(float)):
        s = _doy(y, d0)
        end = on + length if length is not None else off
        lo, hi = max(s, math.ceil(on)), min(s + days - 1, math.floor(end))
        per[y] = max(0, hi - lo + 1)
    yes = [y for y, k in per.items() if k >= m]
    past = {"n_years": len(per), "n_overlap": len(yes), "rate": round(len(yes) / len(per), 3),
            "years_overlap": yes, "years_miss": [y for y, k in per.items() if k < m]}
    return past, per


# M1(2월 말) 예측을 낸다
# 채택됐고 그해 입력이 있을 때만 낸다
# 돌려주는 값은 예측과 잔차 표준편차와 해 수와 입력값이다
def _m1(site, species, year):
    if _M1 is None or _M1X is None:
        return None
    r = _M1[(_M1["수목원"] == site) & (_M1["종"] == species)]
    x = _M1X[(_M1X["수목원"] == site) & (_M1X["연도"] == year)]
    if r.empty or x.empty:
        return None
    r, col = r.iloc[0], r.iloc[0]["입력"]
    xv = x[col].iloc[0]
    if pd.isna(xv):
        return None
    return float(r["평균"]) + float(r["기울기"]) * (float(xv) - float(r["평년_기온"])), float(r["잔차_표준편차"]), \
        int(r["해수"]), float(xv)


# 지도에 찍을 10곳을 돌려준다
# 열은 수목원 위도 경도 관측시작연도 관측종료연도다
def list_sites():
    return _M[["수목원", "수목원_위도", "수목원_경도", "관측시작연도", "관측종료연도"]].rename(
        columns={"수목원_위도": "위도", "수목원_경도": "경도"})


# 그 수목원에서 고를 수 있는 종을 돌려준다
# 열은 종 관측연수 예측가능 추천가능이다
def list_species(site, phenomenon="개화"):
    site, phenomenon = _nfc(site), _nfc(phenomenon)
    bad = _check(site, phenomenon)
    if bad:
        return bad
    pool = SPRING_SPECIES if phenomenon == "개화" else AUTUMN_SPECIES
    rows = []
    for sp in pool:
        r = _row(site, sp, MAIN_STAGE[phenomenon])
        n = 0 if r is None else int(r["관측연수"])
        rows.append({"종": sp, "관측연수": n, "예측가능": n >= MIN_YEARS,
                     "추천가능": phenomenon == "개화" and _hit(site, sp) is not None})
    return pd.DataFrame(rows)


# 최근 5년 편차를 넣었는지에 따라 경고 글을 만든다
def _shift_text(phenomenon, use):
    span, v = RECENT_SPAN.get(phenomenon, ""), RECENT5.get(phenomenon)
    if v is None:
        return []
    if use:
        out = [f"전국 최근 5년({span}) 편차 {v:+.1f}일을 더했습니다." + (" 단풍은 최근 해일수록 늦어지는 흐름이 있고 원인은 확인하지 못했습니다."
                                                              if phenomenon == "단풍" else "")]
        return out + ["범위에는 보정값의 불확실성을 넣지 않았습니다."]
    late = "일렀" if v < 0 else "늦었"
    why = "검증에서 기준을 넘지 못해 예측에 넣지 않았습니다." if FINAL else "이 차이는 아직 예측에 반영하지 않았습니다."
    return [f"최근 5년({span})은 전국 평균으로 {abs(v):.1f}일 {late}습니다. {why}"]


# 따로 다루지 않는 lead 값에 경고를 단다
# 시나리오는 기온 가정이 함께 와야 쓴다
def _lead_warn(lead, temp_scenario):
    if lead == "시나리오" and temp_scenario is None:
        return ["시나리오는 기온 가정(temp_scenario)이 있어야 합니다. 계획 값으로 답합니다."]
    if lead not in LEADS:
        return [f"lead는 {', '.join(LEADS)} 중 하나입니다. 계획 값으로 답합니다."]
    return []


# 기능 1은 예상 날짜와 범위를 낸다
# 계획 값은 조합 평균이다(검증 결과에 따라 최근 5년 보정)
# 여기에 기온 가정을 더할 수 있다
@_safe
def predict_date(site, species, phenomenon="개화", year=2027, lead="계획", level=0.8, temp_scenario=None):
    site, species, phenomenon, lead = _nfc(site), _nfc(species), _nfc(phenomenon), _nfc(lead)
    bad = _check(site, phenomenon, species)
    if bad:
        return bad
    if not _is_int(year) or not YEAR_LO <= year <= YEAR_HI:
        return _fail("BAD_YEAR", f"연도는 {YEAR_LO}~{YEAR_HI} 사이의 정수여야 합니다.")
    year = int(year)
    if isinstance(level, bool) or not isinstance(level, numbers.Real) or level not in T_Q:
        return _fail("BAD_LEVEL", "범위 수준은 0.5, 0.8, 0.9 중 하나여야 합니다.")
    level = float(level)
    sc, bad, warnings = _scenario(temp_scenario, phenomenon)
    if bad:
        return bad
    stage = MAIN_STAGE[phenomenon]
    r = _row(site, species, stage)
    if r is None:
        return _fail("NO_DATA", f"{site}에는 {species}의 {stage} 기록이 없습니다.", n_years=0)
    n = int(r["관측연수"])
    if n < MIN_YEARS:
        return _fail("TOO_FEW_YEARS", f"{site} {species}는 관측한 해가 {n}년뿐이라(기준 {MIN_YEARS}년) 예측하지 않습니다.",
                     n_years=n)
    mu, sd = float(r["평균"]), float(r["표준편차"])
    use = USE_M0R.get(phenomenon, False) and phenomenon in RECENT5
    shift = RECENT5[phenomenon] if use else 0.0
    method = "최근 5년 보정" if use else "기준(과거 평균)"
    shift_warn, extra = _shift_text(phenomenon, use), []
    sd_used, n_used, lead_out = sd, n, "계획"
    if lead == "2월 말":
        if phenomenon != "개화":
            extra.append("단풍에는 2월 말 예측이 없어 계획 값으로 답합니다.")
        elif not FINAL:
            extra.append("임시 구현은 그해 기온을 반영하지 않고 과거 평균으로 답합니다.")
        elif not USE_M1:
            extra.append("2월 기온으로는 예측 오차가 뚜렷이 줄지 않아(검증 결과) 계획 값으로 답합니다.")
        elif sc:
            extra.append("기온 가정과 2월 말 예측은 함께 쓰지 않습니다. 계획 값에 기온 가정을 더했습니다.")
        else:
            got = _m1(site, species, year)
            if got:
                base, sd_used, n_used, xv = got
                shift, method, lead_out, shift_warn = base - mu, "2월 말 예측(M1)", "2월 말", []
                extra.append(f"그해 2월 평균기온 {xv:.1f}도를 넣은 예측입니다. 범위는 모델 잔차로 잡았습니다.")
            else:
                extra.append("그해 2월 기온이 아직 없어 계획 값으로 답합니다.")
    else:
        extra += _lead_warn(lead, temp_scenario)
    warnings += shift_warn + extra
    half = _tq(level, n_used) * sd_used * math.sqrt(1 + 1 / n_used)
    if sc:
        shift += SLOPE * sc["temp_anom"]
        method += " + 기온 가정"
        lead_out = "시나리오"
        warnings.append("범위는 기온을 모를 때의 폭을 그대로 썼습니다.")
    pred = mu + shift
    history, actual = None, None
    if FINAL:
        h = _Y[(_Y["수목원"] == site) & (_Y["종"] == species) & (_Y["현상"] == stage)].sort_values("연도")
        history = pd.DataFrame({"연도": h["연도"].astype(int).to_numpy(), "날짜숫자": h["날짜숫자"].to_numpy(),
                                "날짜": [_iso(int(y), d) for y, d in zip(h["연도"], h["날짜숫자"])],
                                "평균대비_일": (h["날짜숫자"] - mu).round(1).to_numpy()})
        hit = h[h["연도"] == year]
        if len(hit):
            actual = float(hit["날짜숫자"].iloc[0])
            warnings.append("지난 해를 골랐습니다. 예상 날짜는 그해 기록도 들어간 평균으로 낸 값이라 실제보다 잘 맞아 보입니다.")
    out = {"ok": True, "site": site, "species": species, "phenomenon": phenomenon, "stage": stage,
           "year": year, "lead": lead_out, "method": method,
           "pred_doy": round(pred, 1), "pred_date": _iso(year, pred),
           "interval": {"level": level, "low_doy": round(pred - half, 1), "high_doy": round(pred + half, 1),
                        "low_date": _iso(year, pred - half), "high_date": _iso(year, pred + half)},
           "mean_doy": round(mu, 1), "diff_from_mean_days": round(shift, 1), "sd_days": round(sd_used, 2),
           "n_years": n_used, "actual_doy": actual, "history": history, "scenario": sc, "warnings": warnings,
           "source": SOURCE}
    if phenomenon == "개화":
        a, b = _row(site, species, "개화시작"), _row(site, species, "낙화90~100%")
        if a is not None and b is not None and min(a["관측연수"], b["관측연수"]) >= MIN_YEARS:
            # 정제본 판은 꽃 시기를 설계서 5절 1항대로 낸다
            # 개화시작 = 예상 개화50% - 간격 중앙값
            # 낙화 = 개화시작 + 꽃 시기 길이
            if FINAL:
                g, L = _gap(site, species), _length(site, species)[0]
                s = pred - g if g is not None else None
                e = s + L if s is not None and pd.notna(L) else None
            else:
                s, e = float(a["평균"]) + shift, float(b["평균"]) + shift
            if s is not None and e is not None:
                out["bloom_window"] = {"start_doy": round(s, 1), "start_date": _iso(year, s),
                                       "end_doy": round(e, 1), "end_date": _iso(year, e)}
    return out


# 개화시작 분포(평균과 표준편차와 연수)와 꽃 시기 길이를 돌려준다
# 기능 2와 3이 함께 쓴다
# 없으면 실패 dict를 돌려준다
def _onset(site, species):
    r = _row(site, species, "개화시작")
    if r is None:
        return _fail("NO_DATA", f"{site}에는 {species}의 개화시작 기록이 없습니다.", n_years=0)
    n = int(r["관측연수"])
    if n < MIN_YEARS:
        return _fail("TOO_FEW_YEARS", f"{site} {species}는 개화시작을 관측한 해가 {n}년뿐이라(기준 {MIN_YEARS}년) 계산하지 않습니다.",
                     n_years=n)
    length, src = _length(site, species)
    sd = float(r["표준편차"])
    if pd.isna(length) or pd.isna(sd):
        return _fail("NO_DATA", f"{site} {species}는 꽃 시기 길이나 변동 폭을 낼 기록이 없습니다.", n_years=n)
    return {"mu": float(r["평균"]), "sd": sd, "sd_pred": sd * math.sqrt(1 + 1 / n), "n": n,
            "length": length, "length_source": src, "paired": _hit(site, species) is not None}


# 기능 2는 축제 기간이 꽃 시기(개화시작~낙화)와 겹침 기준 일수 이상 겹치는지 본다
@_safe
def festival_overlap(site, species, start_date, days, phenomenon="개화", lead="계획",
                     bloom_length=None, max_days=30, min_overlap_days=1, temp_scenario=None):
    site, species, phenomenon, lead = _nfc(site), _nfc(species), _nfc(phenomenon), _nfc(lead)
    bad = _check(site, phenomenon, species)
    if bad:
        return bad
    if phenomenon != "개화":
        return _fail("NOT_SUPPORTED", "단풍 시기의 시작과 끝을 어느 단계로 볼지 정해지지 않아 아직 계산하지 않습니다.")
    try:
        d0 = _parse(start_date)
    except (ValueError, TypeError):
        return _fail("BAD_DATE", "축제 시작일은 날짜이거나 'YYYY-MM-DD' 형식의 글자여야 합니다.")
    if not YEAR_LO <= d0.year <= YEAR_HI:
        return _fail("BAD_DATE", f"축제 시작일의 연도는 {YEAR_LO}~{YEAR_HI} 사이여야 합니다.")
    if not _is_int(days) or not 1 <= days <= 60:
        return _fail("BAD_DAYS", "축제 일수는 1~60 사이의 정수여야 합니다.")
    if not _is_int(max_days) or not 1 <= max_days <= 60:
        return _fail("BAD_DAYS", "그래프에 그릴 최대 일수는 1~60 사이의 정수여야 합니다.")
    if not _is_int(min_overlap_days) or not 1 <= min_overlap_days <= days:
        return _fail("BAD_DAYS", "겹침 기준 일수는 1 이상이고 축제 일수 이하인 정수여야 합니다.")
    days, max_days, m = int(days), int(max_days), int(min_overlap_days)
    sc, bad, warnings = _scenario(temp_scenario, phenomenon)
    if bad:
        return bad
    # 기능 2는 2월 말 예측을 쓰지 않는다
    if lead == "2월 말":
        warnings.append("기능 2는 2월 말 예측을 쓰지 않아 계획 값으로 답합니다.")
    else:
        warnings += _lead_warn(lead, temp_scenario)
    o = _onset(site, species)
    if "ok" in o:
        return o
    if bloom_length is not None:
        if isinstance(bloom_length, bool) or not isinstance(bloom_length, numbers.Real) or not 1 <= bloom_length <= 60:
            return _fail("BAD_LENGTH", "꽃 시기 길이는 1~60일 사이여야 합니다.")
        length, src = float(bloom_length), "직접 입력"
    else:
        length, src = o["length"], o["length_source"]
    use = FINAL and USE_M0R.get("개화", False) and "개화" in RECENT5
    mu = o["mu"] + (SLOPE * sc["temp_anom"] if sc else 0.0) + (RECENT5["개화"] if use else 0.0)
    sd = o["sd_pred"]
    s0 = d0.timetuple().tm_yday

    def p(s, d):
        return _p_overlap(mu, sd, length, s, d, m)

    starts = list(range(max(1, int(mu) - 45), int(mu) + 46))
    probs = [p(s, days) for s in starts]
    k = max(range(len(starts)), key=lambda i: probs[i])
    by_start = pd.DataFrame({"시작일_날짜숫자": starts, "시작일": [_iso(d0.year, s) for s in starts],
                             "확률": [round(x, 3) for x in probs]})
    ds = list(range(1, max_days + 1))
    by_days = pd.DataFrame({"일수": ds,
                            "확률": [round(p(s0, d), 3) for d in ds],
                            "최적기간_확률": [round(max(p(s, d) for s in starts), 3) for d in ds],
                            "기대겹침일수": [round(_expected_days(mu, sd, length, s0, d), 1) for d in ds]})
    if abs(s0 - mu) > 45:
        warnings.append("입력한 시작일이 이 종의 평균 개화시작에서 45일 넘게 떨어져 있습니다.")
    if m > length + 1:
        warnings.append("겹침 기준 일수가 꽃 시기 길이보다 길어 확률이 0입니다.")
    if not o["paired"] and bloom_length is None:
        warnings.append(f"이 조합은 개화시작과 낙화가 함께 기록된 해가 {MIN_YEARS}년 미만이라 종 중앙값 길이를 썼고 "
                        "수종 추천 순위에는 들어가지 않습니다.")
    prob = p(s0, days)
    head = {"value": round(prob, 3), "basis": "분포", "text": f"겹칠 확률 {_pct_text(prob)}"}
    expected = round(_expected_days(mu, sd, length, s0, days), 1)
    basis, n_years = "과거 분포(정규 근사)", o["n"]
    past, per = _past(site, species, d0, days, m, length if bloom_length is not None else None) if FINAL else (None, None)
    if past is not None and past["n_years"] >= MIN_YEARS and not sc:
        head = {"value": past["rate"], "basis": "과거 해",
                "text": f"{past['n_years']}해 중 {past['n_overlap']}해 겹침"}
        expected = round(sum(per.values()) / len(per), 1)
        basis, n_years = "지난 해 세기", past["n_years"]
    if FINAL:
        warnings.append("prob와 곡선과 가장 좋은 기간은 개화시작 분포와 고정한 꽃 시기 길이로 낸 근사라 "
                        "대표값(지난 해를 센 값)과 조금 다를 수 있습니다.")
        if use:
            warnings.append(f"근사에는 전국 최근 5년 편차 {RECENT5['개화']:+.1f}일을 넣었고 지난 해 세기에는 넣지 않았습니다.")
        if bloom_length is not None and past is not None:
            warnings.append("꽃 시기 길이를 직접 넣어 지난 해 세기에도 그해 개화시작부터 그 길이를 썼습니다.")
        if sc and past is not None:
            warnings.append("기온 가정을 넣어 대표값은 가정을 반영한 분포 값입니다. past는 참고로 그대로 둡니다.")
    warnings.append("가장 좋은 기간은 지난 기록을 보고 고른 것이라 실제보다 좋게 보입니다.")
    return {"ok": True, "site": site, "species": species, "phenomenon": phenomenon,
            "lead": "시나리오" if sc else "계획",
            "start_date": d0.isoformat(), "days": days, "end_date": (d0 + dt.timedelta(days=days - 1)).isoformat(),
            "min_overlap_days": m, "bloom_length_days": length, "bloom_length_source": src,
            "prob": round(prob, 3),
            "headline": head,          # 화면에 크게 보여 줄 값
            "grade": grade(head["value"]),
            "expected_overlap_days": expected,
            "best": {"start_date": _iso(d0.year, starts[k]), "end_date": _iso(d0.year, starts[k] + days - 1),
                     "prob": round(probs[k], 3)},
            "by_days": by_days, "by_start": by_start,
            "past": past,              # 지난 해 중 실제로 겹친 해(정제본 판에만 있음)
            "scenario": sc,
            "basis": basis + (" + 기온 가정" if sc else ""),
            "n_years": n_years, "warnings": warnings, "source": SOURCE}


# 기능 3은 원하는 시기에 맞는 종 순위를 낸다
# 적합도가 높은 순으로 놓는다
# 적합도가 같으면 안정성 점수가 높은 순으로 놓는다
@_safe
def recommend_species(site, start_date, end_date, phenomenon="개화", min_overlap_days=1):
    site, phenomenon = _nfc(site), _nfc(phenomenon)
    bad = _check(site, phenomenon)
    if bad:
        return bad
    if phenomenon != "개화":
        return _fail("NOT_SUPPORTED", "단풍 시기의 시작과 끝을 어느 단계로 볼지 정해지지 않아 아직 계산하지 않습니다.")
    try:
        d0, d1 = _parse(start_date), _parse(end_date)
    except (ValueError, TypeError):
        return _fail("BAD_DATE", "시작일과 종료일은 날짜이거나 'YYYY-MM-DD' 형식의 글자여야 합니다.")
    if d1 < d0 or d0.year != d1.year:
        return _fail("BAD_DATE", "종료일은 시작일과 같은 해의 같은 날 또는 그 뒤여야 합니다.")
    if not YEAR_LO <= d0.year <= YEAR_HI:
        return _fail("BAD_DATE", f"날짜의 연도는 {YEAR_LO}~{YEAR_HI} 사이여야 합니다.")
    s, e = d0.timetuple().tm_yday, d1.timetuple().tm_yday
    days = e - s + 1
    if not _is_int(min_overlap_days) or not 1 <= min_overlap_days <= days:
        return _fail("BAD_DAYS", "겹침 기준 일수는 1 이상이고 원하는 시기의 일수 이하인 정수여야 합니다.")
    m = int(min_overlap_days)
    rows, excluded = [], []
    for sp in SPRING_SPECIES:
        h, o = _hit(site, sp), _onset(site, sp)
        if h is None or "ok" in o:
            excluded.append({"종": sp, "사유": "이 수목원에 기록 없음" if o.get("code") == "NO_DATA"
                             else f"개화시작과 낙화가 함께 기록된 해가 {MIN_YEARS}년 미만"})
            continue
        mu, sd, length = o["mu"], o["sd_pred"], o["length"]       # 기능 2와 같은 분포와 길이
        r50 = _row(site, sp, "개화50%이상")
        sd50 = float(r50["표준편차"]) if r50 is not None and r50["관측연수"] >= MIN_YEARS else None
        recent, past = None, None
        if FINAL:                                                 # 기능 2와 같은 지난 해 세기
            past, per = _past(site, sp, d0, days, m)
            if past is None:
                excluded.append({"종": sp, "사유": "해별 꽃 시기 기록을 찾지 못함"})
                continue
            fit, n_hit = round(past["n_overlap"] / past["n_years"], 2), past["n_overlap"]
            exp_days = round(sum(per.values()) / len(per), 1)
            ry = [y for y in per if LAST_YEAR is not None and y > LAST_YEAR - 5]
            recent = f"{len(ry)}해 중 {sum(per[y] >= m for y in ry)}해" if ry else "기록 없음"
        else:
            fit, n_hit = round(_p_overlap(mu, sd, length, s, days, m), 2), None
            exp_days = round(_expected_days(mu, sd, length, s, days), 1)
        rows.append({"종": sp, "적합도": fit,
                     "안정성점수": stability_score(sd50), "관측연수": int(h["관측연수"]),
                     "평균_개화시작": _iso(d0.year, mu)[5:], "평균_낙화": _iso(d0.year, mu + length)[5:],
                     "변동폭_표준편차": sd50, "개화기간_중앙값": length,
                     "등급": grade(fit),
                     "겹친해수": n_hit,
                     "기대겹침일수": exp_days,
                     "최근5년_겹침": recent})     # 참고 열(순위에 쓰지 않는다)
    if not rows:
        return _fail("NO_CANDIDATE", f"{site}에는 추천할 수 있는 종이 없습니다.", excluded=pd.DataFrame(excluded))
    # 모든 후보의 적합도가 0이면 꽃 철이 아닌 기간이다
    # 적합도 0인 순위표 대신 맞는 종이 없다고 답한다
    if all(r["적합도"] <= 0 for r in rows):
        return _fail("NO_CANDIDATE", f"{site}에는 {d0.isoformat()}~{d1.isoformat()}에 꽃 시기가 {m}일 이상 겹칠 만한 "
                     "종이 없습니다. 기간을 옮기거나 겹침 기준 일수를 줄여 보세요.",
                     excluded=pd.DataFrame(excluded, columns=["종", "사유"]))
    rank = pd.DataFrame(rows).sort_values(["적합도", "안정성점수"], ascending=False, na_position="last")
    rank.insert(0, "순위", range(1, len(rank) + 1))
    warnings = []
    if (rank["적합도"] >= 0.95).sum() >= 2:
        warnings.append("적합도가 1에 가까운 종이 여럿이라 그 안의 순위는 안정성 점수로 갈립니다. 기대겹침일수를 함께 보세요.")
    if FINAL:
        warnings.append("최근5년_겹침은 참고 열이고 순위에는 쓰지 않았습니다.")
    return {"ok": True, "site": site, "phenomenon": phenomenon, "start_date": d0.isoformat(),
            "end_date": d1.isoformat(), "min_overlap_days": m, "ranking": rank.reset_index(drop=True),
            "excluded": pd.DataFrame(excluded, columns=["종", "사유"]), "n_candidates": len(rank),
            "fit_method": "지난 해 세기" if FINAL else "정규 근사(임시)", "warnings": warnings, "source": SOURCE}
