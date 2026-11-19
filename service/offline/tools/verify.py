"""Browser and independent numerical checks. Usage: python tools/verify.py DATA_DIR OUTPUT_DIR"""
import datetime as dt,json,pathlib,sys
import pandas as pd
from playwright.sync_api import sync_playwright
src,out=map(pathlib.Path,sys.argv[1:3]);html=out/'Season Window_실제자료_내부시연.html';results=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    results.append({'check':name,'status':'passed'})
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    ctx=browser.new_context(locale='ko-KR',viewport={'width':1440,'height':1050},accept_downloads=True)
    page=ctx.new_page();errors=[];requests=[];page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url))
    # Test identical document in offline context. Managed Chrome disallows file:// navigation.
    ctx.set_offline(True);page.set_content(html.read_text(),wait_until='load')
    check('offline embedded document initializes',page.locator('#validYears').inner_text()!='—')
    check('no external requests',not requests)
    data=page.evaluate('flowerApp.data')
    check('10 actual markers; 230 region boundaries',page.locator('.site-dot').count()==10 and page.locator('.region').count()==230)
    # Verify all original 256 admin codes were mapped to 230 merged regions in builder output.
    check('administrative inventory',data['quality']['rawAdminCodes']==256 and data['quality']['regionCount']==230)
    # Independent pandas reconstruction of same-tree endpoints from original delivered tree CSV.
    t=pd.read_csv(src/'interim/phenology_tree_year.csv');t=t[~(t['순서오류']|t['첫관측일기록']|t['이상치의심'])]
    pairs=t.pivot(index=['연도','수목원','종','나무번호'],columns='현상',values='날짜숫자')[['개화시작','낙화90~100%']].dropna().reset_index();pairs=pairs[pairs['개화시작']<=pairs['낙화90~100%']]
    expected={}
    for key,g in pairs.groupby(['연도','수목원','종']):
        year,name,sp=key;a=int(g['개화시작'].median()+.5);b=int(g['낙화90~100%'].median()+.5)
        to_ref=lambda doy:dt.datetime(2000,*(dt.datetime(int(year),1,1)+dt.timedelta(days=doy-1)).timetuple()[1:3]).timetuple().tm_yday
        expected[(int(year),name,sp)]=[to_ref(a),to_ref(b),len(g)]
    check('all 1397 site/year bloom windows match independent CSV reconstruction',len(expected)==1397 and all(expected[tuple(r[:3])]==r[3:] for r in data['intervals']))
    names=[s['name'] for s in data['sites']]
    observed=[]
    for name in names:
        page.locator(f'circle[aria-label="{name}"]').dispatch_event('click')
        check(f'map to filter: {name}',page.locator('#place').input_value()==name)
        if name=='국립수목원':
            for sp in ['진달래','왕벚나무']:
                page.select_option('#species',sp);page.select_option('#stage','개화50%이상');observed.append((sp,page.locator('#meanDate').inner_text(),page.locator('#validYears').inner_text()))
    check('species changes produce different observed summaries',observed[0][1:]!=observed[1][1:])
    page.locator('circle[aria-label="국립수목원"]').dispatch_event('click');page.select_option('#species','진달래');page.select_option('#stage','개화50%이상')
    ids=['province','city','place','species','phenomenon','stage','start','end','searchStart','searchEnd'];before=[page.locator('#'+i).input_value() for i in ids]
    colors={}
    for tab in ['analysis','review','plan']:
        page.locator(f'nav button[data-page="{tab}"]').click();check('selection persists across '+tab,[page.locator('#'+i).input_value() for i in ids]==before)
        colors[tab]=page.evaluate("getComputedStyle(document.body).getPropertyValue('--main').trim().toLowerCase()")
        if tab=='review' and 'springReview' not in data:
            check('unconnected review shows no forecast',
                  page.locator('#reviewDate').inner_text()=='—'
                  and '연결되지 않았습니다' in page.locator('#reviewNotice').inner_text())
    check('three requested themes',colors=={'analysis':'#246b50','review':'#a94364','plan':'#b75016'})
    # Compare displayed default results against direct calendar intersections, not JS score().
    windows=[r for r in data['intervals'] if r[1]=='국립수목원' and r[2]=='진달래' and r[0]<2027]
    counts=[];weekends=[]
    for r in windows:
        start=dt.datetime(2000,1,1)+dt.timedelta(days=r[3]-1);end=dt.datetime(2000,1,1)+dt.timedelta(days=r[4]-1)
        a=dt.date(2027,start.month,start.day);b=dt.date(2027,end.month,end.day)
        days=[dt.date(2027,4,3)+dt.timedelta(days=i) for i in range(9)]
        counts.append(sum(a<=d<=b for d in days));weekends.append(sum(a<=d<=b and d.weekday()>=5 for d in days))
    values=page.locator('section.page.plan .metrics .value').all_inner_texts()
    check('actual default days, ratio, weekends independently verified',values==[f'{sum(counts)/len(counts):.1f}일',f'{sum(counts)/len(counts)/9*100:.1f}%',f'{sum(weekends)/len(weekends):.1f}일'])
    chosen=page.locator('#alternatives button').first;nextstart=chosen.get_attribute('data-apply');nextend=chosen.get_attribute('data-end');chosen.click()
    check('apply alternative; duration unchanged and search bounds respected',page.locator('#start').input_value()==nextstart and page.locator('#end').input_value()==nextend and (dt.date.fromisoformat(nextend)-dt.date.fromisoformat(nextstart)).days==8 and '2027-03-15'<=nextstart<=nextend<='2027-04-30')
    # Algebraic invariants over every supported combination and every day in a 9-day moving festival.
    check('all-combination calculation bounds and daily probabilities',page.evaluate('''()=>{const a=flowerApp;for(const s of a.data.sites){for(const sp of a.data.quality.species){const windows=a.data.intervals.filter(r=>r[1]===s.name&&r[2]===sp).map(r=>({a:a.projectDay(r[3],2027),b:a.projectDay(r[4],2027)}));if(!windows.length)continue;for(let start=Date.UTC(2027,1,1);start<Date.UTC(2027,6,1);start+=86400000){const end=start+8*86400000,r=a.score(start,end,windows);if(r.days<0||r.days>9||r.weekend<0||r.weekend>r.days)return false;let sum=0;for(let t=start;t<=end;t+=86400000){const p=a.probability(t,windows);if(p<0||p>1)return false;sum+=p;}if(Math.abs(sum-r.days)>1e-9)return false;}}}return true;}'''))
    check('inclusive endpoints and weekend boundary unit checks',page.evaluate('''()=>{const a=flowerApp,t=Date.UTC(2027,3,3),w=[{a:t,b:t+86400000}];const r=a.score(t,t+86400000,w);return r.days===2&&r.weekend===2&&a.score(t,t,w).days===1;}'''))
    check('past evaluation excludes evaluation and future observations',page.evaluate('flowerApp.selectedIntervals(2015).length>0&&flowerApp.selectedIntervals(2015).every(w=>w.year<2015)'))
    check('review cutoff follows festival year',page.locator('#year').input_value()=='2027' and '2027-02-28' in page.locator('#cutoff').inner_text())
    if 'springReview' in data:
        page.locator('nav button[data-page="review"]').click()
        result=data['springReview']['results']['국립수목원']['진달래']['2027']
        check('spring review renders the embedded result',
              page.locator('#reviewDate').inner_text()==(result['pred_date'] if result['ok'] else '—'))
        if result['ok'] and result['lead']=='계획':
            check('plan fallback reports that target-year weather was unused',
                  '기온 미반영' in page.locator('#reviewNotice').inner_text())
        page.locator('nav button[data-page="plan"]').click()
    check('leap-day projection explicit',page.evaluate("flowerApp.projectDay(60,2027)===Date.UTC(2027,1,28)&&flowerApp.projectDay(60,2028)===Date.UTC(2028,1,29)&&flowerApp.projectDay(61,2027)===Date.UTC(2027,2,1)"))
    page.locator('#start').fill('2027-04-11');page.locator('#end').fill('2027-04-03');page.locator('#end').dispatch_event('change')
    check('invalid festival clears stale results',page.locator('section.page.plan .metrics .value').all_inner_texts()==['—']*3 and page.locator('#calculate').is_disabled())
    page.locator('#start').fill('2027-04-03');page.locator('#end').fill('2027-04-11');page.locator('#searchStart').fill('2027-04-01');page.locator('#searchEnd').fill('2027-04-02');page.locator('#searchEnd').dispatch_event('change')
    check('short search range suppresses alternatives, preserves valid current result',page.locator('#alternatives button').count()==0 and '짧습니다' in page.locator('#warning').inner_text() and page.locator('section.page.plan .metrics .value').first.inner_text()!='—')
    page.select_option('#phenomenon','단풍');check('autumn has observations but no invented viewing interval',page.locator('#calculate').is_disabled() and '단풍 관람 구간' in page.locator('#alternatives').inner_text() and page.locator('#validYears').inner_text()!='—')
    page.select_option('#province','서울특별시');check('unobserved region shows no data, no invented station',page.locator('#place').is_disabled() and page.locator('#validYears').inner_text()=='—' and '관측자료 없음' in page.locator('#dataStatus').inner_text())
    # Find an actual <8-year case from the dataset.
    found=False
    for name in names:
        groups={}
        for r in data['records']:
            if r[1]==name and r[3] in ['개화시작','개화50%이상','낙화90~100%','꽃눈파열'] and not r[6]:groups.setdefault((r[2],r[3]),[]).append(r)
        target=next((k for k,v in groups.items() if 0<len(v)<8),None)
        if target:
            page.locator(f'circle[aria-label="{name}"]').dispatch_event('click');page.select_option('#phenomenon','개화');page.select_option('#species',target[0]);page.select_option('#stage',target[1]);found='자료 부족' in page.locator('#dataStatus').inner_text();break
    check('actual low-sample combination flagged',found)
    page.locator('circle[aria-label="국립수목원"]').dispatch_event('click');page.select_option('#phenomenon','개화');page.select_option('#species','진달래');page.select_option('#stage','개화50%이상')
    page.locator('nav button[data-page="analysis"]').click()
    with page.expect_download() as dl:page.locator('#downloadCsv').click()
    download=dl.value;download.save_as(out/'선택관측요약_검증.csv');check('CSV download has selected observations',len(pd.read_csv(out/'선택관측요약_검증.csv'))==len(page.evaluate('flowerApp.selectedRows()')))
    page.screenshot(path=str(out/'화면_분석.png'),full_page=True)
    page.locator('nav button[data-page="review"]').click();page.screenshot(path=str(out/'화면_재점검.png'),full_page=True)
    page.locator('nav button[data-page="plan"]').click();page.locator('#searchStart').fill('2027-03-15');page.locator('#searchEnd').fill('2027-04-30');page.locator('#searchEnd').dispatch_event('change');page.screenshot(path=str(out/'화면_기간설계.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844});check('mobile has no document horizontal overflow',page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'));page.screenshot(path=str(out/'화면_모바일.png'),full_page=True)
    check('no JavaScript errors',not errors);check('offline throughout, no network requests',not requests)
    browser.close()
report={'passed':len(results),'failed':0,'browser':'system Chromium via Playwright; offline set_content','fileScheme':'Managed Chrome blocks file:// by administrator policy; direct file navigation not validated','checks':results,'defaultExample':{'site':'국립수목원','species':'진달래','festival':'2027-04-03~2027-04-11','values':values}}
(out/'검증결과.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
