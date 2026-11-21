# 설정 모듈 경로와 종 목록과 모델 검증 기준 정의

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SEASON_ROOT") or Path(__file__).resolve().parent.parent)
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
REFERENCE = ROOT / "reference"
TABLES = ROOT / "reports" / "tables"

# 입력
PHENO_SITE = INTERIM / "phenology_site_year.csv"
WEATHER = INTERIM / "weather_daily.csv"
PUBLIC_TA = EXTERNAL / "kma_ta_daily.csv" # 연결 지점 10곳 하루 평균기온(공개 자료)
PUBLIC_TA176 = EXTERNAL / "kma_ta176_daily.csv" # 대구 새 자리 임시 지점 176
STATION_MAP = REFERENCE / "station_map.csv"

# 출력(정제본일 때)
# 줄 단위 관측 파일은 data/processed 아래에만 저장
FEATURES = PROCESSED / "weather_features.csv"
FEATURES_RAW = PROCESSED / "weather_features_daegu_raw.csv"
TRAIN = PROCESSED / "training_table.csv"
MODELS = PROCESSED / "models"
SERVICE = PROCESSED / "service"
PREVIEW = PROCESSED / "preview"

# 식물계절 정제본 파일이 있으면 정제본 반환
# 기상 정제본 파일은 없어도 진행
# 그때는 공개 기온 사용 강수와 일조는 비움
# 식물계절 정제본 파일이 없으면 집계표 반환
def mode():
    return "정제본" if PHENO_SITE.exists() else "집계표"

# 결과 폴더 반환
# features는 줄 단위 변수 저장 위치
# tables는 올리는 집계표 저장 위치
# work는 학습 표와 모델 저장 위치
def folder(kind):
    d = PREVIEW if mode() == "집계표" else {"features": PROCESSED, "tables": TABLES, "work": PROCESSED}[kind]
    d.mkdir(parents=True, exist_ok=True)
    return d

# 수목원별 연결 기상지점 표 반환
def stations():
    m = pd.read_csv(STATION_MAP)
    return m[["수목원", "기상지점", "기상지점명"]].copy()

YEARS = list(range(2009, 2026))
SEED = 20261119

# 종
SPRING_SPECIES = ["왕벚나무", "진달래", "산수유", "미선나무", "백목련", "개나리", "생강나무",
                  "아까시나무", "철쭉", "산철쭉", "마가목"]
EARLY_SPECIES = ["생강나무", "산수유", "미선나무", "개나리", "백목련", "진달래", "왕벚나무"]
LATE_SPECIES = ["산철쭉", "철쭉", "마가목", "아까시나무"]
AUTUMN_SPECIES = ["단풍나무", "당단풍나무", "은행나무", "신갈나무", "졸참나무"]
MIN_YEARS = 8
STAGE_SPRING = "개화50%이상"
STAGE_AUTUMN = "단풍50%이상"
STAGE_START = "개화시작"
STAGE_BUD = "꽃눈파열"
STAGE_END = "낙화90~100%"

# 기상 변수
MISSING_MAX = 0.20
GDD_BASE = 5.0
USABLE_FEB_END = ["기온_12월", "기온_1월", "기온_2월", "기온_1_2월", "기온_겨울_2월말",
                  "강수_1_2월", "일조_1_2월", "누적온도_5도_2월말"]

# 대구(143)는 2015년 6월 11일까지 옛 자리(신암동) 값
# 새 자리 임시 지점 176 자료는 2013년 10월 10일부터 2015년 6월 11일까지
DAEGU = {
    "site": "대구수목원", "old": 143, "new": 176,
    "T0": "2013-10-10", "T1": "2015-06-11",
    # 겹친 610일로 구한 달별 차이(새 자리 - 옛 자리)
    # 176 파일이 없을 때만 사용
    "month_diff": {1: -0.97, 2: -0.87, 3: -0.94, 4: -1.00, 5: -1.03, 6: -1.02,
                   7: -0.83, 8: -0.87, 9: -1.10, 10: -1.01, 11: -0.94, 12: -0.85},
}
DAEGU_ADJUST = True

# 기록에 남길 지점 사건
# 대구만 보정하고 나머지는 표시 또는 설명서 기재
STATION_EVENTS = [
    {"지점": 143, "지점명": "대구", "날짜": "2015-06-12", "크기_도": -0.95, "처리": "보정",
     "내용": "기록이 새 자리(효목동) 값으로 바뀐 날. 그 앞은 새 자리 기준으로 맞춘다"},
    {"지점": 238, "지점명": "금산", "날짜": "2024-08-01", "크기_도": 0.68, "처리": "표시",
     "내용": "지점 이력에 없는 단차. 대아수목원 2025년 봄 줄에 기상주의를 달고 뺀 결과를 함께 적는다"},
    {"지점": 119, "지점명": "수원", "날짜": "2019-07-24", "크기_도": -0.26, "처리": "메모",
     "내용": "서둔동에서 고색동으로 이전. 2~3월로는 -0.18도라 고치지 않는다"},
]
WEATHER_CAUTION = [("대아수목원", 2025)]

# 모델
M1_MAIN = "기온_2월"
M1_SENS = ["기온_1_2월", "기온_겨울_2월말"]
M1_UPPER = "기온_2_3월"
RF_PARAMS = {"n_estimators": 500, "min_samples_leaf": 40, "random_state": SEED, "n_jobs": -1}
RF_WIDE = ["기온_12월", "기온_1월", "기온_2월", "강수_1_2월", "일조_1_2월", "누적온도_5도_2월말"]

# 시계 모델(M2)
# 결과를 보기 전에 확정
M2_START = 32 # 누적온도를 쌓기 시작하는 날짜숫자(2월 1일)
M2_CAP = 212 # 시계 날짜 상한(평년 7월 31일)
M2_EARLY = 7 # 평년 날짜가 기준일보다 이만큼 앞선 계열만 그해 신호로 사용
M2_SERIES = 3 # 계열 통계에 필요한 최소 학습 해 수
M2_SHRINK = 3.0 # 관측 수가 적은 신호를 줄이는 정도
M2_ROWS = 30 # 이득 하나나 폭 배수 하나를 따로 추정하는 데 필요한 최소 학습 줄 수
M2_GAIN = (0.0, 1.5) # 신호별 이득 범위
M2_WIDTH = (0.3, 1.3) # 폭 배수 범위
M2_NEAR = (7, 22) # 꽃 피기 1~3주 전 채점 범위(평년 개화일이 기준일 뒤 7일 초과 22일 이하)
M2_TIE = 1e-6 # 누적온도가 요구량에 이만큼 이내로 모자라면 닿은 것으로 봄

# 검증
LOYO_YEARS = YEARS
FORWARD_YEARS = list(range(2015, 2026))
HOLDOUT_YEARS = [2022, 2023, 2024, 2025]
RECENT_N = 5
BOOT_N = 4000
INTERVAL_LEVEL = 0.8
CLOSE_DAYS = 7
FORWARD_MIN_TRAIN = 3 # 앞으로 가는 검증의 조합 평균에 필요한 최소 학습 해 수

# 채택 규칙
# 결과를 보기 전에 확정(5절 4항)
RULE_MAE = 0.5
RULE_WINS_LOYO = 13 # 한 해씩 빼기 검증 17해 중
RULE_WINS_FORWARD = 9 # 앞으로 가는 검증 11해 중

# 집계표 판에서 빼는 세 점
# A의 수목원 연도 효과 표에서 종이 적어 A가 뺀 점
AGG_DROP = [("경남수목원", 2021), ("금강수목원", 2014), ("대아수목원", 2009)]

# 기준일 이름과 날짜(월과 일)
# 그날까지의 기온만 사용
# 4월 15일은 시계 모델(M2)만 사용
REF_DATES = {"2월 말": None, "3월 15일": (3, 15), "3월 31일": (3, 31), "4월 15일": (4, 15)}
