# HyperFrames 방식 렌더: 멈춘 타임라인을 프레임마다 seek → 스크린샷 → ffmpeg
import asyncio, shutil, sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
from playwright.async_api import async_playwright
FPS = 30
async def main(out_dir, only=None):
    os.makedirs(out_dir, exist_ok=True)
    async with async_playwright() as p:
        chrome = shutil.which('google-chrome') or shutil.which('chrome')
        b = await p.chromium.launch(executable_path=chrome, args=['--no-sandbox', '--font-render-hinting=none'])
        pg = await b.new_page(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1)
        await pg.goto((HERE / 'index.html').as_uri()); await pg.wait_for_timeout(500)
        dur = await pg.evaluate("window.__timelines.main.duration")
        frames = only if only else range(int(dur * FPS))
        el = await pg.query_selector('#root')
        for f in frames:
            await pg.evaluate(f"window.__timelines.main.seek({f / FPS})")
            await el.screenshot(path=f'{out_dir}/{f:05d}.png')
        await b.close()
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        ts = [float(x) for x in sys.argv[2].split(',')]
        asyncio.run(main(str(HERE / 'stills'), [int(t * FPS) for t in ts]))
    else:
        asyncio.run(main(str(HERE / 'frames')))
