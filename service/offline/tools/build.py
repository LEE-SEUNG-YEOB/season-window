"""Build an offline internal demonstration from separately supplied team data.
Optionally embed B-part spring review results at build time.
"""
import argparse, calendar, hashlib, importlib.util, json, os, pathlib, re
from collections import defaultdict
import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('data_dir', type=pathlib.Path)
parser.add_argument('admin_geojson', type=pathlib.Path)
parser.add_argument('output_dir', type=pathlib.Path)
parser.add_argument('--season-root', type=pathlib.Path, help='B파트 저장소 경로')
args = parser.parse_args()
src, admin_path, out = args.data_dir, args.admin_geojson, args.output_dir
out.mkdir(parents=True, exist_ok=True)

def spring_review(root, combinations):
    root = root.resolve()
    module_path = root / 'modeling/season_functions.py'
    if not module_path.is_file():
        parser.error(f'B파트 예측 함수를 찾을 수 없습니다: {module_path}')
    os.environ['SEASON_ROOT'] = str(root)
    spec = importlib.util.spec_from_file_location('season_functions', module_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    fields = (
        'ok', 'message', 'stage', 'pred_date', 'interval', 'method',
        'lead', 'n_years', 'warnings', 'source',
    )
    results = {}
    for name, species in combinations:
        by_year = results.setdefault(name, {}).setdefault(species, {})
        for year in range(model.YEAR_LO, model.YEAR_HI + 1):
            result = model.predict_date(name, species, year=year, lead='2월 말')
            if result.get('code') == 'INTERNAL_ERROR':
                raise RuntimeError(f'{name} / {species} / {year}: {result["message"]}')
            by_year[year] = {key: result[key] for key in fields if key in result}
    return {'firstYear': model.YEAR_LO, 'lastYear': model.YEAR_HI, 'results': results}

admin = json.loads(admin_path.read_text())
groups = defaultdict(list)
for f in admin['features']:
    p=f['properties']; city=re.sub(r'(?<=시).+구$', '', p['sggnm'])
    groups[p['sidonm'],city].append(shape(f['geometry']))
regions=[]; geoms={}
def project(x,y): return [round((x-124.4)*85,2),round((39.1-y)*105,2)]
for (province,city), gs in sorted(groups.items()):
    g=unary_union(gs); geoms[province,city]=g
    simple=g.simplify(.008,preserve_topology=True)
    polygons=list(simple.geoms) if simple.geom_type=='MultiPolygon' else [simple]
    paths=[]
    for p in polygons:
        if p.area < .00006: continue
        coords=[project(*xy) for xy in p.exterior.coords]
        paths.append('M'+'L'.join(f'{x},{y}' for x,y in coords)+'Z')
    regions.append({'province':province,'city':city,'path':''.join(paths)})
site=pd.read_csv(src/'interim/phenology_site_year.csv')
tree=pd.read_csv(src/'interim/phenology_tree_year.csv')
removed=pd.read_csv(src/'interim/phenology_removed.csv')
weather=pd.read_csv(src/'interim/weather_daily.csv')
raws=[]; files=[]
for p in sorted((src/'raw/phenology').glob('*.xlsx')):
    d=pd.read_excel(p);raws.append(d)
    files.append({'file':str(p.relative_to(src)),'rows':len(d),'columns':list(d.columns),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
raw=pd.concat(raws,ignore_index=True)
ex=raw[raw['세부관측'].eq('전시원')]
sites=[]
for name in sorted(site['수목원'].unique()):
    part=ex[ex['관측자'].eq(name)]
    lat,lon=part[['위도','경도']].median()
    matches=[(p,c) for (p,c),g in geoms.items() if g.covers(Point(lon,lat))]
    if len(matches)!=1: raise ValueError(f'Ambiguous administrative link: {name}: {matches}')
    p,c=matches[0]; x,y=project(lon,lat)
    sites.append({'name':name,'province':p,'city':c,'lat':round(lat,6),'lon':round(lon,6),'x':x,'y':y,'coordinateSource':'원본 전시원 기록의 중앙 좌표 · 개별 나무의 정확한 좌표 아님','rawCoordinateRows':len(part)})
# Leap reference calendar (2000) preserves February 29; non-leap years shift March onward.
def normalize(year,doy):
    return float(doy)+(1 if not calendar.isleap(int(year)) and doy>=60 else 0)
assert site.duplicated(['연도','수목원','종','현상']).sum()==0
records=[]
for r in site.to_dict('records'):
    records.append([r['연도'],r['수목원'],r['종'],r['현상'],normalize(r['연도'],r['날짜숫자']),r['나무수'],bool(r['이상치의심'])])
# Pair within the same tree/year, never combine unrelated start/end samples.
clean=tree[~(tree['순서오류']|tree['첫관측일기록']|tree['이상치의심'])]
pivot=clean.pivot(index=['연도','수목원','종','나무번호'],columns='현상',values='날짜숫자')
pairs=pivot[['개화시작','낙화90~100%']].dropna().reset_index()
reverse=int((pairs['낙화90~100%']<pairs['개화시작']).sum())
pairs=pairs[pairs['낙화90~100%']>=pairs['개화시작']]
intervals=[]
for (year,name,species),g in pairs.groupby(['연도','수목원','종']):
    # Round a half-day median explicitly to the later calendar day for daily counts.
    a=int(g['개화시작'].median()+.5); b=int(g['낙화90~100%'].median()+.5)
    intervals.append([int(year),name,species,int(normalize(year,a)),int(normalize(year,b)),len(g)])
quality={
 'rawPhenologyRows':len(raw),'rawInstitutions':sorted(raw['관측자'].unique()),'rawExhibitionRows':len(ex),
 'siteRows':len(site),'treeRows':len(tree),'years':[int(site['연도'].min()),int(site['연도'].max())],
 'sites':len(sites),'species':sorted(site['종'].unique()),'stages':sorted(site['현상'].unique()),
 'removedRows':len(removed),'removedReasons':removed['제외사유'].value_counts().to_dict(),
 'suspectSiteRows':int(site['이상치의심'].sum()),'suspectTreeRows':int(tree['이상치의심'].sum()),
 'orderFlags':int(tree['순서오류'].sum()),'firstObservationFlags':int(tree['첫관측일기록'].sum()),
 'keyDuplicates':int(site.duplicated(['연도','수목원','종','현상']).sum()),'siteNulls':site.isna().sum().to_dict(),
 'weatherRows':len(weather),'weatherStations':int(weather['지점'].nunique()),'weatherDateRange':[weather['일시'].min(),weather['일시'].max()],
 'weatherNulls':weather.isna().sum().to_dict(),'weatherKeyDuplicates':int(weather.duplicated(['지점','일시']).sum()),
 'validPairedTreeYears':len(pairs),'validPairedSiteYears':len(intervals),'reversedPairsExcluded':reverse,
 'regionCount':len(regions),'provinceCount':len({r['province'] for r in regions}),
 'rawAdminCodes':len({f['properties']['sgg'] for f in admin['features']}),
 'adminSHA256':hashlib.sha256(admin_path.read_bytes()).hexdigest(),
 'groupYearDistribution':site.groupby(['수목원','종','현상']).size().value_counts().sort_index().to_dict(),
 'pairYearDistribution':pd.DataFrame(intervals).groupby([1,2]).size().value_counts().sort_index().to_dict(),
 'files':files,
}
for p in sorted((src/'raw/weather').glob('*.csv')):
    d=pd.read_csv(p,encoding='cp949');quality['files'].append({'file':str(p.relative_to(src)),'rows':len(d),'encoding':'cp949','columns':list(d.columns),'dates':[d['일시'].min(),d['일시'].max()]})
for p in sorted((src/'interim').glob('*.csv')):
    d=pd.read_csv(p);quality['files'].append({'file':str(p.relative_to(src)),'rows':len(d),'encoding':'utf-8-sig','columns':list(d.columns),'nulls':d.isna().sum().to_dict(),'exactDuplicates':int(d.duplicated().sum())})
payload={'sites':sites,'regions':regions,'records':records,'intervals':intervals,'quality':quality,'minimumYears':8}
if args.season_root is not None:
    combinations = sorted({(r[1], r[2]) for r in records if r[3] == '개화50%이상'})
    payload['springReview'] = spring_review(args.season_root, combinations)
html=(ROOT/'web.template.html').read_text()
html=html.replace('/*__DATA__*/',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
html=html.replace('/*__APP__*/',(ROOT/'app.js').read_text())
(out/'Season Window_실제자료_내부시연.html').write_text(html,encoding='utf-8')
(out/'데이터_조사결과.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'관측지_연결표.json').write_text(json.dumps(sites,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in quality.items() if k not in ['files','rawInstitutions','species','stages']},ensure_ascii=False,indent=2))
