"""경로와 공통 기준. 모든 분석 스크립트가 이 파일을 불러 쓴다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PHENO = ROOT / "data" / "raw" / "phenology"
RAW_WEATHER = ROOT / "data" / "raw" / "weather"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "reference"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

SPRING_SPECIES = ["왕벚나무", "진달래", "산수유", "미선나무", "백목련", "개나리",
                  "생강나무", "아까시나무", "철쭉", "산철쭉", "마가목"]
AUTUMN_SPECIES = ["단풍나무", "당단풍나무", "은행나무", "신갈나무", "졸참나무"]
CONIFERS = ["소나무", "잣나무", "주목", "구상나무"]


def species_group(name: str) -> str:
    if name in SPRING_SPECIES:
        return "봄꽃"
    if name in AUTUMN_SPECIES:
        return "단풍"
    if name in CONIFERS:
        return "침엽수"
    return "기타"


# 단계 이름: 원본 표기 그대로
SPRING_STAGES = ["꽃눈파열", "개화시작", "개화50%이상", "낙화90~100%",
                 "잎눈파열", "개엽시작", "개엽완료", "비산시작", "비산종료"]
AUTUMN_STAGES = ["단풍시작20~30%", "단풍50%이상", "단풍90~100%", "낙엽90~100%"]
FRUIT_STAGES = ["열매생성", "열매성숙완료"]

# 앞 단계가 뒤 단계보다 늦으면 순서 오류
ORDER_PAIRS = [
    ("꽃눈파열", "개화시작"), ("개화시작", "개화50%이상"),
    ("개화50%이상", "낙화90~100%"), ("개화시작", "낙화90~100%"),
    ("잎눈파열", "개엽시작"), ("개엽시작", "개엽완료"),
    ("단풍시작20~30%", "단풍50%이상"), ("단풍50%이상", "단풍90~100%"),
    ("단풍90~100%", "낙엽90~100%"), ("비산시작", "비산종료"),
]

# 정제 기준값
SITE_SCOPE = "전시원"
MAX_KM_FROM_SITE = 5.0     # 수목원 중앙 좌표에서 이보다 멀면 위치 의심
# 좌표가 멀지만 좌표 입력 오류로 판단해 그대로 쓰는 조합 (근거는 docs/cleaning_rules.md)
COORD_ACCEPT = {("경남수목원", "졸참나무"), ("경남수목원", "신갈나무"), ("물향기수목원", "아까시나무")}
# 그해 첫 관측일(다음 날 포함)에 이미 이 단계로 기록되면 실제 시작은 더 일렀을 수 있다
CENSOR_STAGES = ["개화시작", "개화50%이상", "낙화90~100%", "개엽시작", "개엽완료", "비산시작", "비산종료"]
CENSOR_WINDOW_DAYS = 1
MIN_YEARS = 8              # 변동 폭 계산에 필요한 최소 관측 연수
SPRING_MAX_DOY = 212       # 봄 단계가 7월 31일보다 늦으면 입력 오류
AUTUMN_MIN_DOY = 182       # 가을 단계가 7월 1일보다 이르면 입력 오류
OUTLIER_DAYS = 45          # 같은 수목원·종·단계 중앙값에서 이보다 벗어나면 의심 표시(삭제하지 않음)
