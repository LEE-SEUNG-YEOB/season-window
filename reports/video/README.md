# 처음 화면 장면 영상

`season-window-hero.mp4`: 1920×1080, 30fps, 26초, 소리 없음. 발표 슬라이드나 공모전 제출 자료에 넣을 수 있습니다.

| 시간 | 장면 |
|---|---|
| 0~3.6초 | 제목 "축제는 그대로인데 꽃은 해마다 움직입니다" |
| 3.5~22.4초 | 4월 3일~11일로 고정한 축제 기간 위로 왕벚나무 개화(수목원 전시원 10곳 평균)가 2010년부터 2025년까지 움직임 |
| 22.3~26초 | "8해 / 16해": 축제 9일 내내 꽃이 있었던 해, Season Window |

`index.html`은 HyperFrames 형식의 장면 파일(멈춘 타임라인을 시간으로 넘기는 구조)이고, `render.py`가 프레임마다 화면을 찍어 ffmpeg로 묶습니다.

```
python render.py full
ffmpeg -framerate 30 -i frames/%05d.png -c:v libx264 -crf 18 -pix_fmt yuv420p season-window-hero.mp4
```

들어간 값은 연도별 전국 평균(집계 결과)뿐입니다.
