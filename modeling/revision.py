# 8단계 채택된 기준일의 지난 해 예측을 서비스 표로 저장

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import features as F
import mconfig as C
import models as M

NEAR = "꽃 피기 1~3주 전" # 판정표에 적힌 묶음 채점 이름
FORWARD = M.SCHEMES[1]
OLD = M.spec("M0r", label="기준일에 자름") # 견주는 기준 모델
NEW = M.spec("M2", label="시계와 관측")
KEY = ["수목원", "종", "연도", "기준일"]
SIDES = ("기준", "고친")
COLS = KEY + ["기준일_날짜숫자", "상태", "날짜숫자"] + [f"{m}_{v}" for m in SIDES for v in ("예측", "하한", "상한")]

# 판정표에서 M2를 쓰는 범위 읽기
# 한 채점의 두 검증 줄이 모두 채택이어야 채택
# 채택된 기준일이 있으면 그 기준일에 아직 피지 않은 줄 사용
# 꽃 피기 1~3주 전 채점만 채택이면 기준일마다 평년 개화일이 1~3주 뒤인 줄 사용
# 반환 값은 기준일 목록과 가까운 줄만 쓰는지 여부
def scope(D):
    ok = D.groupby("기준일")["판정"].agg(lambda s: bool((s == "채택").all()))
    refs = [r for r in C.REF_DATES if ok.get(r, False)]
    if refs:
        return refs, False
    return (list(C.REF_DATES), True) if ok.get(NEAR, False) else ([], False)

# 기준 모델과 M2의 줄 단위 예측을 한 줄로 합친 표 생성
# 앞으로 가는 검증 줄만 사용(해마다 그 앞 해들만으로 낸 예측)
# 기준일에 이미 핀 줄 제외
def sheet(P, refs, near):
    use = (P["검증방식"] == FORWARD) & P["기준일"].isin(refs) & ~P["이미핌"].astype(bool)
    if near:
        use &= P["가까운줄"].astype(bool)
    P = P[use]
    a = P.loc[P["spec"] == OLD["id"], KEY + ["상태", "날짜숫자", "pred", "low", "high"]]
    b = P.loc[P["spec"] == NEW["id"], KEY + ["pred", "low", "high"]]
    R = a.merge(b, on=KEY, how="outer", validate="one_to_one", suffixes=SIDES, indicator=True)
    if (R["_merge"] != "both").any():
        sys.exit("!! 기준 모델과 M2의 줄이 서로 다름")
    names = {"pred": "예측", "low": "하한", "high": "상한"}
    R = R.rename(columns={f"{k}{m}": f"{m}_{v}" for m in SIDES for k, v in names.items()})
    R["기준일_날짜숫자"] = [F.refdoy(r, int(y)) for r, y in zip(R["기준일"], R["연도"])]
    return R.sort_values(["수목원", "종", "연도", "기준일_날짜숫자"]).reset_index(drop=True)[COLS]

# 표가 지켜야 할 것을 어긴 줄 수 반환
# 빈 값이 없고 실제 날짜가 기준일 뒤이고 범위가 기준일 뒤에서 아래 가운데 위 순서
def faults(R):
    bad = R[COLS].isna().any(axis=1) | (R["날짜숫자"] <= R["기준일_날짜숫자"])
    for m in SIDES:
        low, mid, high = (R[f"{m}_{v}"] for v in ("하한", "예측", "상한"))
        bad |= (low < R["기준일_날짜숫자"]) | (low > mid) | (mid > high)
    return int(bad.sum())

# 줄 묶음의 줄 수와 평균 오차 계산
# 반환 값은 줄 수와 기준 모델 오차와 M2 오차
def errors(g):
    return len(g), *(float((g[f"{m}_예측"] - g["날짜숫자"]).abs().mean()) for m in SIDES)

# 표의 줄 수와 평균 오차가 7단계 비교표와 다른 채점 이름 반환
# 비교표는 소수 둘째 자리까지라 그만큼의 차이는 같은 값으로 처리
def match(R, T, near):
    T = T[(T["검증방식"] == FORWARD) & (T["종집단"] == "전체") & (T["모델"] == NEW["모델"]) & (T["입력"] == NEW["입력"])
          & (T["기준모델"] == f"{OLD['모델']}({OLD['입력']})")]
    bad = []
    for name, g in ([(NEAR, R)] if near else R.groupby("기준일", sort=False)):
        t = T[T["기준일"] == name]
        n, old, new = errors(g)
        if len(t) != 1 or int(t["줄수"].iloc[0]) != n or abs(float(t["MAE"].iloc[0]) - new) > 0.0051 \
                or abs(float(t["기준대비_MAE차"].iloc[0]) - (new - old)) > 0.0051:
            bad.append(name)
    return bad

def main():
    if C.mode() != "정제본":
        print("정제본(data/interim)이 없어 기준일 예측 표를 만들지 않는다.")
        return
    tdir, mdir = C.folder("tables"), M.folder()
    need = [tdir / "model_challenge_decision.csv", tdir / "model_challenge.csv", mdir / "challenge_predictions.csv"]
    miss = [p.name for p in need if not p.exists()]
    if miss:
        sys.exit(f"!! 7단계 출력이 없음 {miss}")
    refs, near = scope(pd.read_csv(need[0]))
    R = sheet(pd.read_csv(need[2], low_memory=False), refs, near)
    odd, off = faults(R), match(R, pd.read_csv(need[1]), near) if len(R) else []
    if odd or off:
        sys.exit(f"!! 기준일 예측 표 점검 실패: 어긋난 줄 {odd}개 비교표와 다른 채점 {off}")
    out = C.SERVICE / "revision.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out, index=False, encoding="utf-8-sig")
    use = "기준일마다 평년 개화일이 1~3주 뒤인 조합" if near else "기준일에 아직 피지 않은 조합"
    print("M2를 쓰는 기준일:", " ".join(refs) if refs else "없음", "|", use if refs else "기준 모델만 쓴다")
    print(f"저장: {out.relative_to(C.ROOT)} {R.shape}")
    for name, g in sorted(R.groupby("기준일"), key=lambda t: refs.index(t[0])):
        n, old, new = errors(g)
        wide = [float((g[f"{m}_상한"] - g[f"{m}_하한"]).mean()) for m in SIDES]
        print(f"{name}: {n}줄 {int(g['연도'].min())}~{int(g['연도'].max())}년 조합 {g[['수목원', '종']].drop_duplicates().shape[0]}개"
              f" | 평균 오차 기준 {old:.2f}일 고친 {new:.2f}일 | 80% 범위 폭 기준 {wide[0]:.1f}일 고친 {wide[1]:.1f}일")

if __name__ == "__main__":
    main()
