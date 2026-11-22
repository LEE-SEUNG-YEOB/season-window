# 기능 함수 모듈 예상 날짜와 축제 겹침과 종 순위 반환

import datetime as dt
import functools
import math
import numbers
import os
import unicodedata
from pathlib import Path

import pandas as pd

# 저장소 맨 위 폴더 찾기
# 저장소 위치 환경변수가 있으면 그곳 사용
# 없으면 이 파일에서 위로 올라가며 reports/tables가 있는 곳 사용
def root():
    env = os.environ.get("SEASON_ROOT")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "reports" / "tables").is_dir() and (p / "reference").is_dir():
            return p
    raise FileNotFoundError("reports/tables 폴더를 찾지 못했습니다. 저장소 안에서 불러 주세요.")

ROOT = root()
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
# 범위 계산에 쓰는 t 분포 분위수
# 바깥 열쇠는 범위 수준 안쪽 열쇠는 자유도(관측연수 - 1)
# 관측한 해가 적을수록 정규분포보다 넓은 범위
T_Q = {
    0.5: {7: 0.711, 8: 0.706, 9: 0.703, 10: 0.700, 11: 0.697, 12: 0.695, 13: 0.694, 14: 0.692, 15: 0.691, 16: 0.690},
    0.8: {7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372, 11: 1.363, 12: 1.356, 13: 1.350, 14: 1.345, 15: 1.341, 16: 1.337},
    0.9: {7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746},
}
# 등급 기준(임시)
# 첫 값 이상은 높음
# 둘째 값 이상은 보통 그 아래는 낮음
GRADE_CUTS = (0.8, 0.5)
SCENARIO_MAX = 2.0 # 기온 가정을 받는 범위(도)
SCENARIO_WINDOW = "2~3월 평균기온(평년 대비)"
LEADS = ("계획", "2월 말", "시나리오")
YEAR_LO, YEAR_HI = 2009, 2030 # 고를 수 있는 해(첫 관측 해부터 2030년까지)

def table(p):
    return pd.read_csv(p, encoding="utf-8-sig")

_M = table(REFERENCE / "station_map.csv")
SITES = _M["수목원"].tolist()

if FINAL:
    _V = table(SERVICE / "combo_stats.csv") # 수목원 x 종 x 단계별 관측연수와 평균과 표준편차
    _G = table(SERVICE / "gaps.csv")
    _GC = _G[_G["구분"] == "조합"]
    _GS = _G[_G["구분"] == "종"].set_index("종")
    # 개화시작과 낙화가 함께 8년 이상인 조합
    # 집계표 판에서 쓰는 A의 고정일 적중 표와 같은 모양
    _H = _GC[_GC["기간_해수"] >= MIN_YEARS].rename(columns={"기간_해수": "관측연수", "기간_중앙값": "개화기간_중앙값"})
    # 종 중앙값 꽃 시기 길이
    _D = _GS["기간_중앙값"]
    _B = table(SERVICE / "bloom_pairs.csv") # 해별 개화시작과 낙화
    _Y = table(SERVICE / "history.csv") # 해별 날짜
    _S = table(SERVICE / "shift.csv")
    _DEC = table(SERVICE / "decision.csv")
    RECENT5 = {k: float(_S.loc[_S["현상"] == k, "최근5년_편차"].iloc[0]) for k in MAIN_STAGE if (_S["현상"] == k).any()}
    RECENT_SPAN = {r["현상"]: f"{int(r['시작연도'])}~{int(r['끝연도'])}" for _, r in _S.iterrows()}
    LAST_YEAR = int(_B["연도"].max()) if len(_B) else None

    def adoption(item, phen):
        r = _DEC[(_DEC["항목"] == item) & (_DEC["현상"] == phen)]
        return bool(len(r)) and r["판정"].iloc[0] == "채택"

    USE_M0R = {"개화": adoption("M0r", "개화"), "단풍": adoption("M0r", "단풍")}
    USE_M1 = adoption("M1", "개화")
    _M1 = table(SERVICE / "m1.csv") if (SERVICE / "m1.csv").exists() else None
    _M1X = table(SERVICE / "m1_inputs.csv") if (SERVICE / "m1_inputs.csv").exists() else None
    _sl = _DEC[_DEC["항목"] == "가정 입력 기울기"]
    SLOPE = float(_sl["값"].iloc[0]) if len(_sl) and pd.notna(_sl["값"].iloc[0]) else None
else:
    _V = table(TABLES / "variability_site_species.csv") # 수목원 x 종 x 단계별 관측연수와 평균과 표준편차
    _H = table(TABLES / "eda_fixed_day_hit.csv") # 개화시작과 낙화가 함께 8년 이상인 조합 87개
    _D = table(TABLES / "eda_bloom_duration_species.csv").set_index("종")["중앙값"]
    _GC = _GS = _B = _Y = _M1 = _M1X = None
    # 전국 연도별 편차(일) 읽기
    _A = table(TABLES / "eda_year_anomaly.csv").sort_values("연도")
    # 전국 최근 5년 편차(일) 계산
    RECENT5 = {k: float(_A[v].tail(5).mean()) for k, v in MAIN_STAGE.items()}
    _yrs = [int(y) for y in _A["연도"].tail(5)]
    RECENT_SPAN = {k: f"{_yrs[0]}~{_yrs[-1]}" for k in MAIN_STAGE}
    LAST_YEAR = _yrs[-1]
    USE_M0R = {"개화": False, "단풍": True} # 단풍은 설계서 6절 4항대로 최근 5년 보정 적용
    USE_M1 = False
    SLOPE = None
if SLOPE is None: # 2~3월 평균기온 1도당 일수(전국 공통)는 A의 표 값 사용
    SLOPE = float(table(TABLES / "eda_temp_pooled.csv")["기울기_일per도"].iloc[0])
# 8단계가 만든 지난 해 기준일 예측 표 읽기
# 정제본 판이고 파일이 있을 때만 사용
_R = table(SERVICE / "revision.csv") if FINAL and (SERVICE / "revision.csv").exists() else None
# 9단계가 만든 그해 앞 값 표 읽기
# 정제본 판이고 파일이 있을 때만 사용
_P = table(SERVICE / "hindcast.csv") if FINAL and (SERVICE / "hindcast.csv").exists() else None

def error(code, message, **extra):
    return {"ok": False, "code": code, "message": message, "source": SOURCE, "warnings": [], **extra}

# 기능 함수가 예외를 던지지 않게 감싸기
# 예기치 못한 오류는 내부 오류 코드로 반환
def guard(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e: # 화면 멈춤 방지
            return error("INTERNAL_ERROR", "계산 중 예기치 못한 오류가 났습니다.", detail=f"{type(e).__name__}: {e}")
    return inner

def integer(x):
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)

# 맥에서 복사한 한글은 자모가 풀린 꼴로 올 수 있음
# 이름을 찾기 전에 하나로 모은 꼴로 변환
def nfc(x):
    return unicodedata.normalize("NFC", x) if isinstance(x, str) else x

def cdf(x, mu, sd):
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))

def quantile(level, n):
    return T_Q[level][min(max(n - 1, 7), 16)]

# 그해 날짜숫자를 날짜 글자로 변환
# 날짜숫자는 반올림 딱 절반이면 올림
def iso(year, doy):
    return (dt.date(year, 1, 1) + dt.timedelta(days=int(math.floor(doy + 0.5)) - 1)).isoformat()

def day(d):
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))

# d의 월과 일을 그해 날짜숫자로 변환
# 2월 29일은 평년이면 2월 28일로 처리
def yearday(year, d):
    try:
        return dt.date(year, d.month, d.day).timetuple().tm_yday
    except ValueError:
        return dt.date(year, 2, 28).timetuple().tm_yday

def combo(site, species, stage):
    r = _V[(_V["수목원"] == site) & (_V["종"] == species) & (_V["현상"] == stage)]
    return None if r.empty else r.iloc[0]

def hitrate(site, species):
    r = _H[(_H["수목원"] == site) & (_H["종"] == species)]
    return None if r.empty else r.iloc[0]

def issue(site, phenomenon, species=None):
    if not isinstance(site, str) or site not in SITES:
        return error("UNKNOWN_SITE", f"'{site}'은(는) 선택할 수 없는 곳입니다. 가능한 곳: {', '.join(SITES)}")
    if not isinstance(phenomenon, str) or phenomenon not in MAIN_STAGE:
        return error("UNKNOWN_PHENOMENON", "현상은 '개화' 또는 '단풍'이어야 합니다.")
    pool = SPRING_SPECIES if phenomenon == "개화" else AUTUMN_SPECIES
    if species is not None and (not isinstance(species, str) or species not in pool):
        return error("UNKNOWN_SPECIES", f"'{species}'은(는) {phenomenon} 대상 종이 아닙니다. 가능한 종: {', '.join(pool)}")
    return None

# 기온 가정 점검
# 반환 값은 세 가지
# 가정 dict 또는 None과 실패 dict 또는 None과 경고 목록
def scenario(temp_scenario, phenomenon):
    if temp_scenario is None:
        return None, None, []
    if isinstance(temp_scenario, bool) or not isinstance(temp_scenario, numbers.Real) \
            or math.isnan(temp_scenario) or abs(temp_scenario) > SCENARIO_MAX:
        return None, error("BAD_SCENARIO",
                           f"기온 가정은 -{SCENARIO_MAX:g}~+{SCENARIO_MAX:g}도 사이의 숫자여야 합니다."), []
    if phenomenon != "개화":
        return None, None, ["단풍에는 기온 가정을 반영하지 않습니다."]
    x = float(temp_scenario)
    sc = {"temp_anom": x, "window": SCENARIO_WINDOW, "slope_days_per_degree": SLOPE, "shift_days": round(SLOPE * x, 1)}
    return sc, None, [f"기온 가정은 전국 공통 기울기(1도에 {SLOPE:+.1f}일)를 쓴 근사이고 예측이 아닙니다."]

# 표준편차를 안정성 점수로 변환
# 2일 이하는 100점 12일 이상은 0점
# 그 사이는 직선 보간
# 환산식은 설계서에 없어 임시로 정한 값
def stability(sd):
    if sd is None or pd.isna(sd):
        return None
    return int(round(100 * min(max((12 - float(sd)) / 10, 0), 1)))

# 확률이나 적합도를 세 등급으로 변환
# 기준은 임시이며 팀 결정 사항
def grade(value):
    if value is None or pd.isna(value):
        return None
    return "높음" if value >= GRADE_CUTS[0] else ("보통" if value >= GRADE_CUTS[1] else "낮음")

# 화면에 쓸 확률 글자 생성
# 5%p 단위로 표시 양 끝은 뭉갬
def percent(p):
    if p >= 0.95:
        return "95% 이상"
    if p < 0.05:
        return "5% 미만"
    return f"약 {int(round(p * 20)) * 5}%"

# 축제와 꽃 시기가 m일 이상 겹칠 확률 계산
# 축제는 s일부터 s+d-1일까지
# 꽃 시기는 T일부터 T+length일까지
# T는 개화시작이며 정규분포로 근사
def chance(mu, sd, length, s, d, m=1):
    if d < m:
        return 0.0
    return max(0.0, cdf(s + d - m + 0.5, mu, sd) - cdf(s + m - 1 - length - 0.5, mu, sd))

# 축제 d일 중 꽃이 피어 있는 날 수의 기댓값 계산
def expectation(mu, sd, length, s, d):
    return sum(cdf(k + 0.5, mu, sd) - cdf(k - length - 0.5, mu, sd) for k in range(s, s + d))

# 꽃 시기 길이(낙화 - 개화시작)와 출처 반환
# 함께 기록된 해가 8년 이상이면 조합 중앙값 사용
# 아니면 종 중앙값 사용
def duration(site, species):
    h = hitrate(site, species)
    if h is not None:
        return float(h["개화기간_중앙값"]), f"조합 중앙값({int(h['관측연수'])}년)"
    return (float(_D[species]) if species in _D.index else float("nan")), "종 중앙값"

# 개화시작에서 개화50%이상까지의 간격 중앙값 반환(정제본 판)
# 8년 미만이면 종 중앙값 사용
# 없으면 None 반환
def gap(site, species):
    r = _GC[(_GC["수목원"] == site) & (_GC["종"] == species)]
    if len(r) and r["간격_해수"].iloc[0] >= MIN_YEARS:
        return float(r["간격_중앙값"].iloc[0])
    if species in _GS.index and pd.notna(_GS.loc[species, "간격_중앙값"]):
        return float(_GS.loc[species, "간격_중앙값"])
    return None

# 지난 해마다 축제 기간과 그해 꽃 시기가 겹친 날 수 세기(정제본 판)
# 축제 날짜는 월과 일을 그해 날짜숫자로 바꿔 맞춤
# length를 주면 꽃 시기 끝은 개화시작 + length
# 반환 값은 past dict와 해별 겹친 날 수 dict
# 기록이 없으면 None 두 개 반환
def record(site, species, d0, days, m, length=None):
    if _B is None:
        return None, None
    P = _B[(_B["수목원"] == site) & (_B["종"] == species)].sort_values("연도")
    if P.empty:
        return None, None
    per = {}
    for y, on, off in zip(P["연도"].astype(int), P["개화시작"].astype(float), P["낙화"].astype(float)):
        s = yearday(y, d0)
        end = on + length if length is not None else off
        lo, hi = max(s, math.ceil(on)), min(s + days - 1, math.floor(end))
        per[y] = max(0, hi - lo + 1)
    yes = [y for y, k in per.items() if k >= m]
    past = {"n_years": len(per), "n_overlap": len(yes), "rate": round(len(yes) / len(per), 3),
            "years_overlap": yes, "years_miss": [y for y, k in per.items() if k < m]}
    return past, per

# M1(2월 말) 예측 계산
# 채택됐고 그해 입력이 있을 때만 계산
# 반환 값은 예측과 잔차 표준편차와 해 수와 입력값
def line(site, species, year):
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

# 지도에 찍을 10곳 반환
# 열은 수목원 위도 경도 관측시작연도 관측종료연도
def sites():
    return _M[["수목원", "수목원_위도", "수목원_경도", "관측시작연도", "관측종료연도"]].rename(
        columns={"수목원_위도": "위도", "수목원_경도": "경도"})

# 그 수목원에서 고를 수 있는 종 반환
# 열은 종 관측연수 예측가능 추천가능
def catalog(site, phenomenon="개화"):
    site, phenomenon = nfc(site), nfc(phenomenon)
    bad = issue(site, phenomenon)
    if bad:
        return bad
    pool = SPRING_SPECIES if phenomenon == "개화" else AUTUMN_SPECIES
    rows = []
    for sp in pool:
        r = combo(site, sp, MAIN_STAGE[phenomenon])
        n = 0 if r is None else int(r["관측연수"])
        rows.append({"종": sp, "관측연수": n, "예측가능": n >= MIN_YEARS,
                     "추천가능": phenomenon == "개화" and hitrate(site, sp) is not None})
    return pd.DataFrame(rows)

# 최근 5년 편차 적용 여부에 따른 경고 글 생성
def note(phenomenon, use):
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

# 따로 다루지 않는 lead 값에 경고 추가
# 시나리오는 기온 가정이 함께 와야 사용
def warning(lead, temp_scenario):
    if lead == "시나리오" and temp_scenario is None:
        return ["시나리오는 기온 가정(temp_scenario)이 있어야 합니다. 계획 값으로 답합니다."]
    if lead not in LEADS:
        return [f"lead는 {', '.join(LEADS)} 중 하나입니다. 계획 값으로 답합니다."]
    return []

# 기능 1 예상 날짜와 범위 반환
# 계획 값은 조합 평균(검증 결과에 따라 최근 5년 보정)
# 여기에 기온 가정 추가 가능
@guard
def forecast(site, species, phenomenon="개화", year=2027, lead="계획", level=0.8, temp_scenario=None):
    site, species, phenomenon, lead = nfc(site), nfc(species), nfc(phenomenon), nfc(lead)
    bad = issue(site, phenomenon, species)
    if bad:
        return bad
    if not integer(year) or not YEAR_LO <= year <= YEAR_HI:
        return error("BAD_YEAR", f"연도는 {YEAR_LO}~{YEAR_HI} 사이의 정수여야 합니다.")
    year = int(year)
    if isinstance(level, bool) or not isinstance(level, numbers.Real) or level not in T_Q:
        return error("BAD_LEVEL", "범위 수준은 0.5, 0.8, 0.9 중 하나여야 합니다.")
    level = float(level)
    sc, bad, warnings = scenario(temp_scenario, phenomenon)
    if bad:
        return bad
    stage = MAIN_STAGE[phenomenon]
    r = combo(site, species, stage)
    if r is None:
        return error("NO_DATA", f"{site}에는 {species}의 {stage} 기록이 없습니다.", n_years=0)
    n = int(r["관측연수"])
    if n < MIN_YEARS:
        return error("TOO_FEW_YEARS", f"{site} {species}는 관측한 해가 {n}년뿐이라(기준 {MIN_YEARS}년) 예측하지 않습니다.",
                     n_years=n)
    mu, sd = float(r["평균"]), float(r["표준편차"])
    use = USE_M0R.get(phenomenon, False) and phenomenon in RECENT5
    shift = RECENT5[phenomenon] if use else 0.0
    method = "최근 5년 보정" if use else "기준(과거 평균)"
    shift_warn, extra = note(phenomenon, use), []
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
            got = line(site, species, year)
            if got:
                base, sd_used, n_used, xv = got
                shift, method, lead_out, shift_warn = base - mu, "2월 말 예측(M1)", "2월 말", []
                extra.append(f"그해 2월 평균기온 {xv:.1f}도를 넣은 예측입니다. 범위는 모델 잔차로 잡았습니다.")
            else:
                extra.append("그해 2월 기온이 아직 없어 계획 값으로 답합니다.")
    else:
        extra += warning(lead, temp_scenario)
    warnings += shift_warn + extra
    half = quantile(level, n_used) * sd_used * math.sqrt(1 + 1 / n_used)
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
                                "날짜": [iso(int(y), d) for y, d in zip(h["연도"], h["날짜숫자"])],
                                "평균대비_일": (h["날짜숫자"] - mu).round(1).to_numpy()})
        hit = h[h["연도"] == year]
        if len(hit):
            actual = float(hit["날짜숫자"].iloc[0])
            warnings.append("지난 해를 골랐습니다. 예상 날짜는 그해 기록도 들어간 평균으로 낸 값이라 실제보다 잘 맞아 보입니다.")
    out = {"ok": True, "site": site, "species": species, "phenomenon": phenomenon, "stage": stage,
           "year": year, "lead": lead_out, "method": method,
           "pred_doy": round(pred, 1), "pred_date": iso(year, pred),
           "interval": {"level": level, "low_doy": round(pred - half, 1), "high_doy": round(pred + half, 1),
                        "low_date": iso(year, pred - half), "high_date": iso(year, pred + half)},
           "mean_doy": round(mu, 1), "diff_from_mean_days": round(shift, 1), "sd_days": round(sd_used, 2),
           "n_years": n_used, "actual_doy": actual, "history": history, "scenario": sc, "warnings": warnings,
           "source": SOURCE}
    if phenomenon == "개화":
        a, b = combo(site, species, "개화시작"), combo(site, species, "낙화90~100%")
        if a is not None and b is not None and min(a["관측연수"], b["관측연수"]) >= MIN_YEARS:
            # 정제본 판은 꽃 시기를 설계서 5절 1항대로 계산
            # 개화시작 = 예상 개화50% - 간격 중앙값
            # 낙화 = 개화시작 + 꽃 시기 길이
            if FINAL:
                g, L = gap(site, species), duration(site, species)[0]
                s = pred - g if g is not None else None
                e = s + L if s is not None and pd.notna(L) else None
            else:
                s, e = float(a["평균"]) + shift, float(b["평균"]) + shift
            if s is not None and e is not None:
                out["bloom_window"] = {"start_doy": round(s, 1), "start_date": iso(year, s),
                                       "end_doy": round(e, 1), "end_date": iso(year, e)}
    return out

# 개화시작 분포(평균과 표준편차와 연수)와 꽃 시기 길이 반환
# 기능 2와 3이 함께 사용
# 없으면 실패 dict 반환
def onset(site, species):
    r = combo(site, species, "개화시작")
    if r is None:
        return error("NO_DATA", f"{site}에는 {species}의 개화시작 기록이 없습니다.", n_years=0)
    n = int(r["관측연수"])
    if n < MIN_YEARS:
        return error("TOO_FEW_YEARS", f"{site} {species}는 개화시작을 관측한 해가 {n}년뿐이라(기준 {MIN_YEARS}년) 계산하지 않습니다.",
                     n_years=n)
    length, src = duration(site, species)
    sd = float(r["표준편차"])
    if pd.isna(length) or pd.isna(sd):
        return error("NO_DATA", f"{site} {species}는 꽃 시기 길이나 변동 폭을 낼 기록이 없습니다.", n_years=n)
    return {"mu": float(r["평균"]), "sd": sd, "sd_pred": sd * math.sqrt(1 + 1 / n), "n": n,
            "length": length, "length_source": src, "paired": hitrate(site, species) is not None}

# 기능 2 축제 기간이 꽃 시기(개화시작~낙화)와 기준 일수 이상 겹치는지 확인
@guard
def overlap(site, species, start_date, days, phenomenon="개화", lead="계획",
                     bloom_length=None, max_days=30, min_overlap_days=1, temp_scenario=None):
    site, species, phenomenon, lead = nfc(site), nfc(species), nfc(phenomenon), nfc(lead)
    bad = issue(site, phenomenon, species)
    if bad:
        return bad
    if phenomenon != "개화":
        return error("NOT_SUPPORTED", "단풍 시기의 시작과 끝을 어느 단계로 볼지 정해지지 않아 아직 계산하지 않습니다.")
    try:
        d0 = day(start_date)
    except (ValueError, TypeError):
        return error("BAD_DATE", "축제 시작일은 날짜이거나 'YYYY-MM-DD' 형식의 글자여야 합니다.")
    if not YEAR_LO <= d0.year <= YEAR_HI:
        return error("BAD_DATE", f"축제 시작일의 연도는 {YEAR_LO}~{YEAR_HI} 사이여야 합니다.")
    if not integer(days) or not 1 <= days <= 60:
        return error("BAD_DAYS", "축제 일수는 1~60 사이의 정수여야 합니다.")
    if not integer(max_days) or not 1 <= max_days <= 60:
        return error("BAD_DAYS", "그래프에 그릴 최대 일수는 1~60 사이의 정수여야 합니다.")
    if not integer(min_overlap_days) or not 1 <= min_overlap_days <= days:
        return error("BAD_DAYS", "겹침 기준 일수는 1 이상이고 축제 일수 이하인 정수여야 합니다.")
    days, max_days, m = int(days), int(max_days), int(min_overlap_days)
    sc, bad, warnings = scenario(temp_scenario, phenomenon)
    if bad:
        return bad
    # 기능 2는 2월 말 예측 미사용
    if lead == "2월 말":
        warnings.append("기능 2는 2월 말 예측을 쓰지 않아 계획 값으로 답합니다.")
    else:
        warnings += warning(lead, temp_scenario)
    o = onset(site, species)
    if "ok" in o:
        return o
    if bloom_length is not None:
        if isinstance(bloom_length, bool) or not isinstance(bloom_length, numbers.Real) or not 1 <= bloom_length <= 60:
            return error("BAD_LENGTH", "꽃 시기 길이는 1~60일 사이여야 합니다.")
        length, src = float(bloom_length), "직접 입력"
    else:
        length, src = o["length"], o["length_source"]
    use = FINAL and USE_M0R.get("개화", False) and "개화" in RECENT5
    mu = o["mu"] + (SLOPE * sc["temp_anom"] if sc else 0.0) + (RECENT5["개화"] if use else 0.0)
    sd = o["sd_pred"]
    s0 = d0.timetuple().tm_yday

    def p(s, d):
        return chance(mu, sd, length, s, d, m)

    starts = list(range(max(1, int(mu) - 45), int(mu) + 46))
    probs = [p(s, days) for s in starts]
    k = max(range(len(starts)), key=lambda i: probs[i])
    by_start = pd.DataFrame({"시작일_날짜숫자": starts, "시작일": [iso(d0.year, s) for s in starts],
                             "확률": [round(x, 3) for x in probs]})
    ds = list(range(1, max_days + 1))
    by_days = pd.DataFrame({"일수": ds,
                            "확률": [round(p(s0, d), 3) for d in ds],
                            "최적기간_확률": [round(max(p(s, d) for s in starts), 3) for d in ds],
                            "기대겹침일수": [round(expectation(mu, sd, length, s0, d), 1) for d in ds]})
    if abs(s0 - mu) > 45:
        warnings.append("입력한 시작일이 이 종의 평균 개화시작에서 45일 넘게 떨어져 있습니다.")
    if m > length + 1:
        warnings.append("겹침 기준 일수가 꽃 시기 길이보다 길어 확률이 0입니다.")
    if not o["paired"] and bloom_length is None:
        warnings.append(f"이 조합은 개화시작과 낙화가 함께 기록된 해가 {MIN_YEARS}년 미만이라 종 중앙값 길이를 썼고 "
                        "수종 추천 순위에는 들어가지 않습니다.")
    prob = p(s0, days)
    head = {"value": round(prob, 3), "basis": "분포", "text": f"겹칠 확률 {percent(prob)}"}
    expected = round(expectation(mu, sd, length, s0, days), 1)
    basis, n_years = "과거 분포(정규 근사)", o["n"]
    past, per = record(site, species, d0, days, m, length if bloom_length is not None else None) if FINAL else (None, None)
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
            "headline": head, # 화면에 크게 보여 줄 값
            "grade": grade(head["value"]),
            "expected_overlap_days": expected,
            "best": {"start_date": iso(d0.year, starts[k]), "end_date": iso(d0.year, starts[k] + days - 1),
                     "prob": round(probs[k], 3)},
            "by_days": by_days, "by_start": by_start,
            "past": past, # 지난 해 중 실제로 겹친 해(정제본 판에만 있음)
            "scenario": sc,
            "basis": basis + (" + 기온 가정" if sc else ""),
            "n_years": n_years, "warnings": warnings, "source": SOURCE}

# 기능 3 원하는 시기에 맞는 종 순위 반환
# 적합도가 높은 순으로 정렬
# 적합도가 같으면 안정성 점수가 높은 순으로 정렬
@guard
def ranking(site, start_date, end_date, phenomenon="개화", min_overlap_days=1):
    site, phenomenon = nfc(site), nfc(phenomenon)
    bad = issue(site, phenomenon)
    if bad:
        return bad
    if phenomenon != "개화":
        return error("NOT_SUPPORTED", "단풍 시기의 시작과 끝을 어느 단계로 볼지 정해지지 않아 아직 계산하지 않습니다.")
    try:
        d0, d1 = day(start_date), day(end_date)
    except (ValueError, TypeError):
        return error("BAD_DATE", "시작일과 종료일은 날짜이거나 'YYYY-MM-DD' 형식의 글자여야 합니다.")
    if d1 < d0 or d0.year != d1.year:
        return error("BAD_DATE", "종료일은 시작일과 같은 해의 같은 날 또는 그 뒤여야 합니다.")
    if not YEAR_LO <= d0.year <= YEAR_HI:
        return error("BAD_DATE", f"날짜의 연도는 {YEAR_LO}~{YEAR_HI} 사이여야 합니다.")
    s, e = d0.timetuple().tm_yday, d1.timetuple().tm_yday
    days = e - s + 1
    if not integer(min_overlap_days) or not 1 <= min_overlap_days <= days:
        return error("BAD_DAYS", "겹침 기준 일수는 1 이상이고 원하는 시기의 일수 이하인 정수여야 합니다.")
    m = int(min_overlap_days)
    rows, excluded = [], []
    for sp in SPRING_SPECIES:
        h, o = hitrate(site, sp), onset(site, sp)
        if h is None or "ok" in o:
            excluded.append({"종": sp, "사유": "이 수목원에 기록 없음" if o.get("code") == "NO_DATA"
                             else f"개화시작과 낙화가 함께 기록된 해가 {MIN_YEARS}년 미만"})
            continue
        mu, sd, length = o["mu"], o["sd_pred"], o["length"] # 기능 2와 같은 분포와 길이
        r50 = combo(site, sp, "개화50%이상")
        sd50 = float(r50["표준편차"]) if r50 is not None and r50["관측연수"] >= MIN_YEARS else None
        recent, past = None, None
        if FINAL: # 기능 2와 같은 지난 해 세기
            past, per = record(site, sp, d0, days, m)
            if past is None:
                excluded.append({"종": sp, "사유": "해별 꽃 시기 기록을 찾지 못함"})
                continue
            fit, n_hit = round(past["n_overlap"] / past["n_years"], 2), past["n_overlap"]
            exp_days = round(sum(per.values()) / len(per), 1)
            ry = [y for y in per if LAST_YEAR is not None and y > LAST_YEAR - 5]
            recent = f"{len(ry)}해 중 {sum(per[y] >= m for y in ry)}해" if ry else "기록 없음"
        else:
            fit, n_hit = round(chance(mu, sd, length, s, days, m), 2), None
            exp_days = round(expectation(mu, sd, length, s, days), 1)
        rows.append({"종": sp, "적합도": fit,
                     "안정성점수": stability(sd50), "관측연수": int(h["관측연수"]),
                     "평균_개화시작": iso(d0.year, mu)[5:], "평균_낙화": iso(d0.year, mu + length)[5:],
                     "변동폭_표준편차": sd50, "개화기간_중앙값": length,
                     "등급": grade(fit),
                     "겹친해수": n_hit,
                     "기대겹침일수": exp_days,
                     "최근5년_겹침": recent}) # 참고 열(순위에 미사용)
    if not rows:
        return error("NO_CANDIDATE", f"{site}에는 추천할 수 있는 종이 없습니다.", excluded=pd.DataFrame(excluded))
    # 모든 후보의 적합도가 0이면 꽃 철이 아닌 기간
    # 적합도 0인 순위표 대신 맞는 종 없음으로 응답
    if all(r["적합도"] <= 0 for r in rows):
        return error("NO_CANDIDATE", f"{site}에는 {d0.isoformat()}~{d1.isoformat()}에 꽃 시기가 {m}일 이상 겹칠 만한 "
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

# 기준일 예측 한 줄에서 한쪽 모델의 값을 화면용으로 변환
def step(r, side, method):
    year = int(r["연도"])
    pred, low, high = (float(r[f"{side}_{v}"]) for v in ("예측", "하한", "상한"))
    return {"method": method, "pred_doy": round(pred, 1), "pred_date": iso(year, pred),
            "interval": {"level": 0.8, "low_doy": round(low, 1), "high_doy": round(high, 1),
                         "low_date": iso(year, low), "high_date": iso(year, high)},
            "error_days": round(abs(pred - float(r["날짜숫자"])), 1)}

# 기능 4 지난 해를 기준일로 되돌린 예측 반환
# 그해 기준일에 아직 피지 않았던 조합이 대상
# 기준 모델과 시계 모델(M2)의 예측과 80% 범위와 실제 날짜 반환
# 두 예측 모두 그 앞 해들의 기록만으로 계산(앞으로 가는 검증)
@guard
def revision(site, species, year):
    site, species = nfc(site), nfc(species)
    bad = issue(site, "개화", species)
    if bad:
        return bad
    if not integer(year) or not YEAR_LO <= year <= YEAR_HI:
        return error("BAD_YEAR", f"연도는 {YEAR_LO}~{YEAR_HI} 사이의 정수여야 합니다.")
    year = int(year)
    if _R is None or _R.empty:
        return error("NOT_SUPPORTED", "기준일 예측 표(revision.csv)가 없거나 비어 있어 지난 해를 되돌려 볼 수 없습니다.")
    q = _R[(_R["수목원"] == site) & (_R["종"] == species)]
    years = sorted({int(y) for y in q["연도"]})
    q = q[q["연도"] == year].sort_values("기준일_날짜숫자")
    if q.empty:
        h = _Y[(_Y["수목원"] == site) & (_Y["종"] == species) & (_Y["현상"] == MAIN_STAGE["개화"]) & (_Y["연도"] == year)]
        d = _R.loc[_R["연도"] == year, "기준일_날짜숫자"]
        early = len(h) > 0 and len(d) > 0 and float(h["날짜숫자"].iloc[0]) <= float(d.max())
        why = "그해 기준일에는 이미 피어 있었습니다." if early else "그해는 되돌려 볼 예측이 없습니다."
        can = f"되돌려 볼 수 있는 해: {', '.join(map(str, years))}" if years else "이 조합은 되돌려 볼 수 있는 해가 없습니다."
        return error("NO_DATA", f"{site} {species} {year}년: {why} {can}", years=years)
    r = q.iloc[-1] # 채택된 기준일이 여럿이면 가장 늦은 기준일 사용
    rd, actual = int(r["기준일_날짜숫자"]), float(r["날짜숫자"])
    return {"ok": True, "site": site, "species": species, "phenomenon": "개화", "stage": MAIN_STAGE["개화"],
            "year": year, "ref": r["기준일"], "ref_doy": rd, "ref_date": iso(year, rd), "state": r["상태"],
            "base": step(r, "기준", "기준 모델(최근 5년 보정)"), "revised": step(r, "고친", "시계 모델(M2)"),
            "actual_doy": actual, "actual_date": iso(year, actual), "years": years,
            "warnings": ["두 예측 모두 그 앞 해들의 기록만으로 낸 값입니다.",
                         "기준 모델도 기준일까지 피지 않았다는 사실을 넣은 값입니다."],
            "source": SOURCE}

# 기간이 꽃 시기와 겹친 날 수 계산
def bloom(s, days, on, off):
    lo, hi = max(s, math.ceil(on)), min(s + days - 1, math.floor(off))
    return max(0, hi - lo + 1)

# 겹칠 확률이 가장 높은 시작 날짜숫자와 그 확률 반환
# 찾는 범위와 고르는 순서는 기능 2와 같음
def peak(mu, sd, length, days, m):
    starts = list(range(max(1, int(mu) - 45), int(mu) + 46))
    probs = [chance(mu, sd, length, s, days, m) for s in starts]
    k = max(range(len(starts)), key=lambda i: probs[i])
    return starts[k], probs[k]

# 기능 5 지난 해마다 그해 앞 기록만으로 기능 2를 다시 돌린 값과 실제 반환
# 앞 해가 8해 이상인 해가 대상
# 입력한 기간과 그해 가장 좋은 기간과 작년 꽃 시기에 맞춘 기간을 그해 실제 꽃 시기와 견줌
# 조합 하나는 해 수가 적어 비율이 아니라 센 값으로 반환
@guard
def hindcast(site, species, start_date, days, min_overlap_days=1):
    site, species = nfc(site), nfc(species)
    bad = issue(site, "개화", species)
    if bad:
        return bad
    try:
        d0 = day(start_date)
    except (ValueError, TypeError):
        return error("BAD_DATE", "축제 시작일은 날짜이거나 'YYYY-MM-DD' 형식의 글자여야 합니다.")
    if not YEAR_LO <= d0.year <= YEAR_HI:
        return error("BAD_DATE", f"축제 시작일의 연도는 {YEAR_LO}~{YEAR_HI} 사이여야 합니다.")
    if not integer(days) or not 1 <= days <= 60:
        return error("BAD_DAYS", "축제 일수는 1~60 사이의 정수여야 합니다.")
    if not integer(min_overlap_days) or not 1 <= min_overlap_days <= days:
        return error("BAD_DAYS", "겹침 기준 일수는 1 이상이고 축제 일수 이하인 정수여야 합니다.")
    days, m = int(days), int(min_overlap_days)
    if _P is None or _P.empty:
        return error("NOT_SUPPORTED", "되돌려 보기 표(hindcast.csv)가 없거나 비어 있어 지난 해를 되돌려 볼 수 없습니다.")
    T = _P[(_P["수목원"] == site) & (_P["종"] == species) & (_P["앞해수"] >= MIN_YEARS)].sort_values("연도")
    if T.empty:
        return error("NO_DATA", f"{site} {species}는 앞선 기록이 {MIN_YEARS}해 이상인 해가 없어 되돌려 볼 수 없습니다.", years=[])
    P = _B[(_B["수목원"] == site) & (_B["종"] == species)]
    own = {int(y): (float(on), float(off)) for y, on, off in zip(P["연도"], P["개화시작"], P["낙화"])}
    use = USE_M0R.get("개화", False)

    # 그해 s일에 시작하는 기간이 말한 값과 그해 실제 겹친 날 수 반환
    # 말한 값은 앞선 짝 기록이 8해 이상이면 지난 해 세기이고 아니면 식의 확률
    def view(y, s, prob):
        d = dt.date(y, 1, 1) + dt.timedelta(days=s - 1)
        before = [k for k in own if k < y]
        yes = [k for k in before if bloom(yearday(k, d), days, *own[k]) >= m]
        head = {"value": round(prob, 3), "basis": "분포", "text": f"겹칠 확률 {percent(prob)}"}
        if len(before) >= MIN_YEARS:
            head = {"value": round(len(yes) / len(before), 3), "basis": "과거 해",
                    "text": f"{len(before)}해 중 {len(yes)}해 겹침"}
        got = bloom(s, days, *own[y])
        return {"start_date": iso(y, s), "end_date": iso(y, s + days - 1), "prob": round(prob, 3), "headline": head,
                "grade": grade(head["value"]), "overlap_days": got, "hit": got >= m}

    def total(rows):
        yes = sum(1 for x in rows if x["hit"])
        return {"n_years": len(rows), "n_overlap": yes, "text": f"{len(rows)}해 중 {yes}해 겹침"}

    track = []
    for r in T.to_dict("records"):
        y, n = int(r["연도"]), int(r["앞해수"])
        mu = float(r["평균"]) + (float(r["최근5년_편차"]) if use else 0.0)
        sd, length = float(r["표준편차"]) * math.sqrt(1 + 1 / n), float(r["길이"])
        s = yearday(y, d0)
        top, best = peak(mu, sd, length, days, m)
        row = {"year": y, "n_prior": n, **view(y, s, chance(mu, sd, length, s, days, m)),
               "bloom": {"start_date": iso(y, own[y][0]), "end_date": iso(y, own[y][1])},
               "best": view(y, top, best), "last": None}
        prev = [k for k in own if k < y]
        if prev:
            # 가장 가까운 앞 해의 꽃 시기 한가운데에 맞춘 기간을 달과 날로 그해에 옮김
            on, off = own[max(prev)]
            mid = dt.date(max(prev), 1, 1) + dt.timedelta(days=math.floor((on + off) / 2 - (days - 1) / 2 + 0.5) - 1)
            s = yearday(y, mid)
            got = bloom(s, days, *own[y])
            row["last"] = {"year": max(prev), "start_date": iso(y, s), "end_date": iso(y, s + days - 1),
                           "overlap_days": got, "hit": got >= m}
        track.append(row)
    warnings = ["해마다 그해 앞까지의 기록만으로 다시 계산한 값입니다.",
                "조합 하나는 해 수가 적어 센 값으로만 보여 줍니다. 검증된 확률이 아닙니다."]
    if use:
        warnings.append("확률과 가장 좋은 기간에는 그해 앞까지의 전국 최근 5년 편차를 넣었고 지난 해 세기에는 넣지 않았습니다.")
        warnings.append(f"최근 5년 편차를 쓰기로 한 결정은 {LAST_YEAR}년까지의 기록으로 내렸고 이 결정까지 되돌린 것은 아닙니다.")
    return {"ok": True, "site": site, "species": species, "start_date": d0.isoformat(), "days": days,
            "min_overlap_days": m, "years": [t["year"] for t in track], "track": track,
            "summary": {"input": total(track), "best": total([t["best"] for t in track]),
                        "last": total([t["last"] for t in track if t["last"]])},
            "warnings": warnings, "source": SOURCE}

# C 화면이 부르는 옛 이름 유지
list_sites = sites
list_species = catalog
predict_date = forecast
festival_overlap = overlap
recommend_species = ranking
