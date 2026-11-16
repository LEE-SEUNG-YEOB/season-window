"""식물계절 xlsx 20개를 합치고 정제한다.

출력 (모두 data/interim/, 저장소에 올리지 않음)
- phenology_tree_year.csv : 나무 한 그루 x 연도 x 단계 = 한 줄
- phenology_site_year.csv : 수목원 x 종 x 연도 x 단계 = 한 줄 (B, C 전달용)
- phenology_removed.csv   : 제외한 기록과 사유
출력 (reports/tables/, 집계)
- cleaning_summary.csv    : 정제 단계별 줄 수
처리 기준 설명은 docs/cleaning_rules.md
"""
import numpy as np
import pandas as pd
import config as C

steps = []          # (단계, 남은 수, 제외 수, 단위)
removed = []


def log(name, kept, dropped, unit):
    steps.append({"단계": name, "남은수": kept, "제외수": dropped, "단위": unit})


# 1. 병합 --------------------------------------------------------------
frames = []
for f in sorted(C.RAW_PHENO.glob("*.xlsx")):
    df = pd.read_excel(f, dtype=str)
    df["원본파일"] = f.stem
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
assert len(frames) == 20, len(frames)
assert (raw["종명"] == raw["원본파일"]).all()
log("20개 파일 병합", len(raw), 0, "줄")

raw["관측일"] = pd.to_datetime(raw["관측일자"])
raw["연도"] = raw["연도"].astype(int)
raw["위도"] = raw["위도"].astype(float)
raw["경도"] = raw["경도"].astype(float)


def drop(df, mask, reason):
    out = df[mask].copy()
    out["제외사유"] = reason
    removed.append(out)
    return df[~mask].copy()


# 2. 완전 중복 줄 ------------------------------------------------------
key_cols = [c for c in raw.columns if c not in ("No.",)]
d = raw.duplicated(subset=key_cols, keep="first")
df = drop(raw, d, "완전 중복 줄")
log("완전 중복 줄 제거", len(df), int(d.sum()), "줄")

# 3. 연도와 관측일자의 연도가 다른 줄 ----------------------------------
m = df["관측일"].dt.year != df["연도"]
df = drop(df, m, "연도 칸과 관측일자의 연도 불일치")
log("연도 불일치 제거", len(df), int(m.sum()), "줄")

# 4. 전시원만 ----------------------------------------------------------
m = df["세부관측"] != C.SITE_SCOPE
n_out = int(m.sum())
df = df[~m].copy()          # 범위 밖은 오류가 아니므로 제외 목록에 넣지 않는다
log("전시원 10곳만 사용", len(df), n_out, "줄")

# 5. 좌표: 수목원 중앙 좌표에서 먼 줄 ----------------------------------
med = df.groupby("관측자")[["위도", "경도"]].median().rename(
    columns={"위도": "중앙위도", "경도": "중앙경도"})
df = df.join(med, on="관측자")
df["중앙거리_km"] = np.hypot(
    (df["위도"] - df["중앙위도"]) * 111.0,
    (df["경도"] - df["중앙경도"]) * 111.0 * np.cos(np.radians(df["위도"])))
m = df["중앙거리_km"] > C.MAX_KM_FROM_SITE
df = drop(df, m, f"좌표가 수목원 중앙에서 {C.MAX_KM_FROM_SITE:.0f}km 이상")
log("좌표 의심 줄 제거", len(df), int(m.sum()), "줄")

# 6. 긴 형태: 한 줄 = 나무 x 날짜 x 단계 --------------------------------
df["날짜숫자"] = df["관측일"].dt.dayofyear
long = df.melt(
    id_vars=["No.", "관측자", "종명", "성상", "번호", "연도", "관측일", "날짜숫자"],
    value_vars=["꽃", "잎", "열매"], var_name="구분", value_name="현상",
).dropna(subset=["현상"])
long = long.rename(columns={"관측자": "수목원", "종명": "종", "번호": "나무번호"})
log("단계 기록으로 펼침", len(long), 0, "기록")

# 7. 계절 범위 밖 ------------------------------------------------------
m = (long["현상"].isin(C.SPRING_STAGES) & (long["날짜숫자"] > C.SPRING_MAX_DOY)) | \
    (long["현상"].isin(C.AUTUMN_STAGES) & (long["날짜숫자"] < C.AUTUMN_MIN_DOY))
long = drop(long, m, "계절 범위 밖 (봄 단계가 8월 이후 또는 가을 단계가 6월 이전)")
log("계절 범위 밖 제거", len(long), int(m.sum()), "기록")

# 8. 같은 나무·같은 해·같은 단계가 여러 번: 가장 이른 날짜 ---------------
long = long.sort_values(["수목원", "종", "나무번호", "연도", "현상", "관측일"])
k = ["수목원", "종", "나무번호", "연도", "현상"]
m = long.duplicated(subset=k, keep="first")
long = drop(long, m, "같은 나무·해·단계의 반복 기록 (가장 이른 날짜만 남김)")
log("반복 기록 정리", len(long), int(m.sum()), "기록")

# 9. 순서 오류 표시 ----------------------------------------------------
wide = long.pivot(index=["수목원", "종", "나무번호", "연도"], columns="현상", values="날짜숫자")
bad = set()
pair_counts = []
for a, b in C.ORDER_PAIRS:
    v = wide.index[wide[a] > wide[b]]
    pair_counts.append({"앞단계": a, "뒷단계": b, "비교가능": int((wide[a].notna() & wide[b].notna()).sum()),
                        "순서오류": len(v)})
    for idx in v:
        bad.add(idx + (a,))
        bad.add(idx + (b,))
long["순서오류"] = [t in bad for t in zip(long["수목원"], long["종"], long["나무번호"], long["연도"], long["현상"])]

# 10. 이상치 의심 표시 (삭제하지 않음) ----------------------------------
g = long.groupby(["수목원", "종", "현상"])["날짜숫자"]
long["중앙값차"] = long["날짜숫자"] - g.transform("median")
long["이상치의심"] = long["중앙값차"].abs() > C.OUTLIER_DAYS

long["종구분"] = long["종"].map(C.species_group)
tree = long[["연도", "수목원", "종", "종구분", "나무번호", "구분", "현상", "관측일", "날짜숫자",
             "순서오류", "이상치의심"]].sort_values(["수목원", "종", "나무번호", "연도", "날짜숫자"])
tree["관측일"] = tree["관측일"].dt.strftime("%Y-%m-%d")
log("나무 단위 정제본", len(tree), 0, "기록")

# 11. 수목원 단위 대표값: 순서 오류를 뺀 나무들의 중앙값 -----------------
ok = tree[~tree["순서오류"]]
site = ok.groupby(["연도", "수목원", "종", "현상"]).agg(
    날짜숫자=("날짜숫자", "median"), 나무수=("나무번호", "nunique"),
    이상치의심=("이상치의심", "any")).reset_index()
site["종구분"] = site["종"].map(C.species_group)
site = site[["연도", "수목원", "종", "현상", "날짜숫자", "나무수", "종구분", "이상치의심"]]
log("수목원 단위 대표값 (순서 오류 제외, 나무 중앙값)", len(site), int(tree["순서오류"].sum()), "기록")

# 저장 ------------------------------------------------------------------
C.INTERIM.mkdir(parents=True, exist_ok=True)
C.TABLES.mkdir(parents=True, exist_ok=True)
tree.to_csv(C.INTERIM / "phenology_tree_year.csv", index=False, encoding="utf-8-sig")
site.to_csv(C.INTERIM / "phenology_site_year.csv", index=False, encoding="utf-8-sig")
rem = pd.concat(removed, ignore_index=True)
rem.to_csv(C.INTERIM / "phenology_removed.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(steps).to_csv(C.TABLES / "cleaning_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(pair_counts).to_csv(C.TABLES / "cleaning_order_errors.csv", index=False, encoding="utf-8-sig")

print(pd.DataFrame(steps).to_string(index=False))
print(pd.DataFrame(pair_counts).to_string(index=False))
print("이상치 의심:", int(tree["이상치의심"].sum()), "/ 순서오류 표시:", int(tree["순서오류"].sum()))
print(rem["제외사유"].value_counts().to_string())

# 개화시작이 빠진 경우 현황
fl = wide[wide[["개화시작", "개화50%이상", "낙화90~100%"]].notna().any(axis=1)]
miss = fl["개화시작"].isna() & fl["개화50%이상"].notna()
gap = (wide["개화50%이상"] - wide["개화시작"]).dropna()
print(f"꽃 기록이 있는 나무·해 {len(fl)}, 개화시작 없이 개화50%만 있음 {int(miss.sum())}, "
      f"개화시작→개화50% 간격 중앙값 {gap.median():.0f}일")
