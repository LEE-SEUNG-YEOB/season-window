'use strict';
const DATA=JSON.parse(document.getElementById('embedded-data').textContent);
const $=id=>document.getElementById(id), DAY=86400000;
const escapeHtml=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const date=v=>v?new Date(v+'T00:00:00Z'):null;
const iso=t=>new Date(t).toISOString().slice(0,10);
const md=d=>iso(Date.UTC(2000,0,Math.round(d))).slice(5).replace('-','/');
const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
const sd=a=>a.length>1?Math.sqrt(a.reduce((s,x)=>s+(x-avg(a))**2,0)/(a.length-1)):null;
const fmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—';
const table=(headers,rows)=>`<table><thead><tr>${headers.map(x=>`<th>${escapeHtml(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const stageFamilies={개화:['개화시작','개화50%이상','낙화90~100%','꽃눈파열'],단풍:['단풍시작20~30%','단풍50%이상','단풍90~100%','낙엽90~100%']};
let currentPage='plan';
function options(id,values,preferred){const el=$(id),old=preferred??el.value;el.replaceChildren(...values.map(x=>new Option(x,x)));if(values.includes(old))el.value=old;el.disabled=!values.length;}
function selectedSite(){return DATA.sites.find(s=>s.name===$('place').value);}
function baseRows(){return DATA.records.filter(r=>r[1]===$('place').value);}
function selectedRows(species=$('species').value){return baseRows().filter(r=>r[2]===species&&r[3]===$('stage').value&&($('includeSuspect').checked||!r[6])).sort((a,b)=>a[0]-b[0]);}
function loadCities(preferred){options('city',DATA.regions.filter(r=>r.province===$('province').value).map(r=>r.city),preferred);loadPlaces();}
function loadPlaces(preferred){const list=DATA.sites.filter(s=>s.province===$('province').value&&s.city===$('city').value).map(s=>s.name);options('place',list,preferred);if(!list.length)$('place').add(new Option('이 지역의 정제 관측자료 없음',''));loadSpecies();}
function loadSpecies(){const allowed=stageFamilies[$('phenomenon').value];options('species',[...new Set(baseRows().filter(r=>allowed.includes(r[3])).map(r=>r[2]))].sort((a,b)=>a.localeCompare(b,'ko')));loadStages();}
function loadStages(){options('stage',stageFamilies[$('phenomenon').value].filter(stage=>baseRows().some(r=>r[2]===$('species').value&&r[3]===stage)));update();}
function focusRegion(){const p=$('map').querySelector('.region.selected');if(p){const b=p.getBBox(),w=Math.max(150,b.width+35),h=Math.max(160,b.height+35);$('map').setAttribute('viewBox',`${b.x+b.width/2-w/2} ${b.y+b.height/2-h/2} ${w} ${h}`);}}
function drawMap(){
 const ns='http://www.w3.org/2000/svg', svg=$('map');svg.replaceChildren();
 function element(tag,attrs){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e;}
 for(const r of DATA.regions){const p=element('path',{d:r.path,class:'region'+(r.province===$('province').value&&r.city===$('city').value?' selected':''),tabindex:0,role:'button','aria-label':`${r.province} ${r.city}`});const t=element('title',{});t.textContent=`${r.province} ${r.city}`;p.append(t);const choose=()=>{$('province').value=r.province;loadCities(r.city);focusRegion();};p.addEventListener('click',choose);p.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});svg.append(p);}

 for(const s of DATA.sites){const g=element('g',{});const c=element('circle',{cx:s.x,cy:s.y,r:6,class:'site-dot'+(s.name===$('place').value?' selected':''),tabindex:0,role:'button','aria-label':s.name});const t=element('title',{});t.textContent=`${s.name} · ${s.province} ${s.city}`;c.append(t);const choose=()=>{$('province').value=s.province;options('city',DATA.regions.filter(r=>r.province===s.province).map(r=>r.city),s.city);loadPlaces(s.name);focusRegion();};c.addEventListener('click',choose);c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});g.append(c);
 // Labels only for selected site; all markers retain accessible names/tooltips.
 if(s.name===$('place').value){const label=element('text',{x:s.x+9,y:s.y-9,class:'map-label'});label.textContent=s.name;g.append(label);}svg.append(g);}
 const note=element('text',{x:22,y:34,class:'map-label'});note.textContent='전국 관측지 · 전시원 10곳';svg.append(note);
}
function observations(rows){if(!rows.length)return '<p class="error">선택 조합의 유효 관측자료가 없습니다.</p>';const lo=Math.floor(Math.min(...rows.map(r=>r[4]))-7),hi=Math.ceil(Math.max(...rows.map(r=>r[4]))+7),mean=avg(rows.map(r=>r[4]));const y=d=>230-(d-lo)/(hi-lo)*190;const x=i=>60+i/Math.max(1,rows.length-1)*650;
 return `<svg class="chartsvg" viewBox="0 0 760 280" role="img" aria-label="연도별 ${escapeHtml($('stage').value)} 관측일. 표에 정확한 값 제공"><line x1="50" y1="${y(mean)}" x2="730" y2="${y(mean)}" stroke="var(--accent)" stroke-dasharray="5 4"/><text x="54" y="${y(mean)-8}" font-size="12">평균 ${md(mean)}</text><polyline points="${rows.map((r,i)=>`${x(i)},${y(r[4])}`).join(' ')}" fill="none" stroke="var(--main)" stroke-width="2"/>${rows.map((r,i)=>`<circle cx="${x(i)}" cy="${y(r[4])}" r="4" fill="var(--main)"><title>${r[0]}년 ${md(r[4])}${r[6]?' · 이상치 의심':''}</title></circle><text x="${x(i)}" y="260" text-anchor="middle" font-size="10">${r[0]}</text>`).join('')}<text x="2" y="35" font-size="11">${md(hi)}</text><text x="2" y="232" font-size="11">${md(lo)}</text></svg>`+table(['연도','관측일(월/일)','윤년 기준 환산 DOY','대표값에 사용된 나무 수','품질'],rows.map(r=>[r[0],md(r[4]),fmt(r[4],1),r[5],r[6]?'<span class="low">이상치 의심</span>':'표시 없음']));
}
function analysis(){const rows=selectedRows(), site=selectedSite(), raw=baseRows().filter(r=>r[2]===$('species').value&&r[3]===$('stage').value), vals=rows.map(r=>r[4]);
 $('validYears').textContent=rows.length?`${rows.length}년`:'—';$('meanDate').textContent=rows.length?md(avg(vals)):'—';
 $('dataStatus').textContent=!site?'관측자료 없음':rows.length<DATA.minimumYears?`자료 부족 · ${rows.length}년 / 기준 8년`:`유효 관측 ${rows.length}년 · 과거 기록`;
 $('dataStatus').classList.toggle('error',!site||rows.length<DATA.minimumYears);
 $('stageMeaning').textContent=$('stage').value?`${$('stage').value}: 제공 파일의 관측 단계명입니다. 기관별 판정 원문은 아직 확인하지 못했습니다.`:'지원되는 관측 단계가 없습니다.';
 $('observations').innerHTML=observations(rows);
 const species=[...new Set(baseRows().filter(r=>r[3]===$('stage').value).map(r=>r[2]))];
 const stats=species.map(s=>{const a=selectedRows(s).map(r=>r[4]);return {s,a,n:a.length,sd:sd(a)};}).sort((a,b)=>(b.n>=8)-(a.n>=8)||(a.sd??Infinity)-(b.sd??Infinity));
 $('stability').innerHTML=table(['수종','평균 월/일','표준편차(일)','최소~최대','유효 연수','비교 상태'],stats.map(s=>[escapeHtml(s.s),s.n?md(avg(s.a)):'—',fmt(s.sd),s.n?`${md(Math.min(...s.a))}~${md(Math.max(...s.a))}`:'—',s.n,s.n>=8?'8년 기준 충족':'<span class="low">표본 부족</span>']));
 const q=DATA.quality;
 $('quality').innerHTML=`<p>선택 단계 ${raw.length}행 중 의심 기록 ${raw.filter(r=>r[6]).length}행. 현재 분석 ${rows.length}년.</p><p>2009~2025년의 17년을 기준으로 미관측/현재 제외 ${17-rows.length}년입니다. 원자료의 누락 원인까지 판별한 수치는 아닙니다.</p><p>날짜 변동 폭(표본 표준편차): ${fmt(sd(vals))}일<br>최소~최대 차이: ${vals.length?fmt(Math.max(...vals)-Math.min(...vals)):'—'}일</p><p>전체 정제본 ${q.siteRows.toLocaleString()}행 · 키 중복 ${q.keyDuplicates} · 빈 셀 0 · 의심 표시 ${q.suspectSiteRows}행.</p><p>나무 단위: 순서오류 ${q.orderFlags}건, 첫관측일기록 ${q.firstObservationFlags}건, 의심 ${q.suspectTreeRows}건(서로 겹칠 수 있음). 원자료 단계 미기록은 정상 빈칸일 수도 있어 정제본 빈 셀과 구분합니다.</p><p>확인된 사실: 원본 기관 11곳 중 정제본은 10곳이며, 개화50%이상은 ${DATA.records.filter(r=>r[3]==='개화50%이상').length.toLocaleString()}행입니다. A 메모의 통계 검정·기온 효과는 여기서 재검증하지 않았습니다.</p>`;
 $('downloadCsv').disabled=!rows.length;
}
function validateDates(){const a=date($('start').value),b=date($('end').value),lo=date($('searchStart').value),hi=date($('searchEnd').value);if(!a||!b||isNaN(+a)||isNaN(+b)||b<a)return {error:'시작일과 종료일을 확인해 주세요.'};const n=Math.round((b-a)/DAY)+1;
 if(a.getUTCFullYear()!==b.getUTCFullYear()||n>366)return {error:'한 해 안의 축제 기간만 지원합니다.'};
 let searchError='';if(!lo||!hi||isNaN(+lo)||isNaN(+hi)||hi<lo)searchError='변경 가능한 기간을 확인해 주세요.';else if(lo.getUTCFullYear()!==a.getUTCFullYear()||hi.getUTCFullYear()!==a.getUTCFullYear())searchError='변경 범위는 축제와 같은 연도 안이어야 합니다.';else if((hi-lo)/DAY+1<n)searchError='변경 가능한 기간이 축제 일수보다 짧습니다.';
 return {a:+a,b:+b,lo:+lo,hi:+hi,n,year:a.getUTCFullYear(),searchError};}
function projectDay(day,year){const ref=new Date(Date.UTC(2000,0,day));const last=new Date(Date.UTC(year,ref.getUTCMonth()+1,0)).getUTCDate();return Date.UTC(year,ref.getUTCMonth(),Math.min(last,ref.getUTCDate()));}
function selectedIntervals(year){return DATA.intervals.filter(r=>r[1]===$('place').value&&r[2]===$('species').value&&r[0]<year).map(r=>({year:r[0],a:projectDay(r[3],year),b:projectDay(r[4],year),start:r[3],end:r[4],trees:r[5]}));}
function score(a,b,windows){let days=0,weekend=0;const details=[];for(const w of windows){let d=0,week=0;for(let t=Math.max(a,w.a);t<=Math.min(b,w.b);t+=DAY){d++;if([0,6].includes(new Date(t).getUTCDay()))week++;}days+=d;weekend+=week;details.push({...w,days:d,weekend:week});}return {days:windows.length?days/windows.length:0,weekend:windows.length?weekend/windows.length:0,details};}
function probability(t,windows){return windows.filter(w=>w.a<=t&&w.b>=t).length/windows.length;}
function curve(v,windows){const lo=Math.min(v.a-14*DAY,v.searchError?v.a:v.lo),hi=Math.max(v.b+14*DAY,v.searchError?v.b:v.hi),n=(hi-lo)/DAY;const x=t=>50+(t-lo)/(hi-lo)*670,y=p=>220-p*180;let points=[];for(let t=lo;t<=hi;t+=DAY)points.push(`${x(t)},${y(probability(t,windows))}`);
 const plot=document.querySelector('.page.plan .plot');plot.style.height='auto';plot.style.border='0';plot.style.background='white';plot.innerHTML=`<svg class="chartsvg" viewBox="0 0 760 280" role="img" aria-label="날짜별 기록상 개화 구간 포함 비율과 선택 축제 기간"><rect x="${x(v.a)}" y="40" width="${Math.max(1,x(v.b+DAY)-x(v.a))}" height="180" fill="var(--soft)"/>${[0,.5,1].map(p=>`<line x1="50" x2="735" y1="${y(p)}" y2="${y(p)}" stroke="#dfe5df"/><text x="5" y="${y(p)+4}" font-size="11">${p*100}%</text>`).join('')}<polyline points="${points.join(' ')}" fill="none" stroke="var(--main)" stroke-width="2.5"/>${[0,.25,.5,.75,1].map(f=>`<text x="${50+670*f}" y="250" text-anchor="middle" font-size="11">${iso(lo+Math.round(n*f)*DAY).slice(5)}</text>`).join('')}</svg>`;
 $('lower').textContent=iso(lo);$('upper').textContent=iso(hi);
}
function clearPlan(reason){document.querySelectorAll('section.page.plan .metrics .value').forEach(el=>el.textContent='—');$('alternatives').innerHTML=`<tr><td colspan="5" class="error">${escapeHtml(reason)}</td></tr>`;$('yearDetails').textContent=reason;const plot=document.querySelector('.page.plan .plot');plot.style.height='auto';plot.innerHTML=`<p class="error">${escapeHtml(reason)}</p>`;$('lower').textContent='';$('upper').textContent='';$('curveNote').textContent='계산 결과 없음';}
function plan(){const v=validateDates();$('warning').className='error';$('warning').textContent=v.error||v.searchError||'';$('duration').textContent=v.error?'기간을 확인해 주세요.':`시작일·종료일 포함 ${v.n}일`;
 let reason=v.error||(!$('place').value?'이 지역의 정제 관측자료가 없습니다.':$('phenomenon').value!=='개화'?'단풍 관람 구간은 정의 미확정입니다. 관측일 분석은 사용할 수 있습니다.':'');const windows=v.error?[]:selectedIntervals(v.year);if(!reason&&windows.length<8)reason=`자료 부족: 같은 나무의 시작·낙화가 있는 유효 구간 ${windows.length}년 / 기준 8년. 축제 연도보다 이전 자료만 사용합니다.`;
 $('calculate').disabled=!!reason;if(reason){clearPlan(reason);return;}
 const result=score(v.a,v.b,windows),values=[`${fmt(result.days)}일`,`${fmt(result.days/v.n*100)}%`,`${fmt(result.weekend)}일`];document.querySelectorAll('section.page.plan .metrics .value').forEach((el,i)=>el.textContent=values[i]);
 $('curveNote').textContent=`유효 구간 ${windows.length}년(${Math.min(...windows.map(w=>w.year))}~${Math.max(...windows.map(w=>w.year))}) · 과거 구간을 ${v.year}년 달력에 투영. 음영은 선택 축제 일정이며 곡선은 기록상 구간 포함 비율입니다. 미래 예측 확률이 아닙니다.`;curve(v,windows);
 const alternatives=[];if(!v.searchError)for(let a=v.lo;a+(v.n-1)*DAY<=v.hi;a+=DAY){const b=a+(v.n-1)*DAY;if(a===v.a)continue;alternatives.push({a,b,...score(a,b,windows)});}alternatives.sort((a,b)=>b.days-a.days||b.weekend-a.weekend||a.a-b.a);
 const row=(name,a,b,r,button)=>`<tr><td>${name}</td><td>${iso(a)} ~ ${iso(b)}</td><td>${fmt(r.days)}일 (${fmt(r.days/v.n*100)}%)</td><td>${fmt(r.weekend)}일</td><td>${button?`<button data-apply="${iso(a)}" data-end="${iso(b)}">적용</button>`:'현재 일정'}</td></tr>`;
 $('alternatives').innerHTML=row('현재 선택',v.a,v.b,result,false)+(v.searchError?`<tr><td colspan="5" class="error">${escapeHtml(v.searchError)}</td></tr>`:alternatives.slice(0,3).map((r,i)=>row(`대안 ${i+1}`,r.a,r.b,r,true)).join(''));
 $('yearDetails').innerHTML=table(['관측 연도','시작~낙화 기록 월/일','유효 나무 쌍','겹친 일수','주말 일수(축제 연도 달력)'],result.details.map(w=>[w.year,`${md(w.start)} ~ ${md(w.end)}`,w.trees,w.days,w.weekend]));
}
function review(){
 const v=validateDates(), festival=date($('start').value);
 const year=festival&&!isNaN(+festival)?festival.getUTCFullYear():2027;
 options('year',[String(year)],String(year));$('year').disabled=true;
 $('cutoff').textContent=`기상정보 기준일: ${iso(new Date(Date.UTC(year,2,0)))}`;
 $('reviewOverlap').textContent='—';
 $('reviewDate').textContent='—';
 $('reviewDetails').textContent='';
 $('reviewWarnings').textContent='';
 $('reviewWarnings').classList.add('hidden');
 const notice=$('reviewNotice');
 if($('phenomenon').value==='단풍'){
  notice.textContent='봄꽃 전용 화면입니다. 단풍 예측은 제공하지 않습니다.';
  return;
 }
 if(!v.error){
  const windows=selectedIntervals(v.year);
  if(windows.length>=8)$('reviewOverlap').textContent=`${fmt(score(v.a,v.b,windows).days)}일`;
 }
 const reviewData=DATA.springReview, site=selectedSite();
 if(!reviewData){
  notice.textContent='봄꽃 재점검 결과가 연결되지 않았습니다.';
  return;
 }
 if(year<reviewData.firstYear||year>reviewData.lastYear){
  notice.textContent=`재점검 지원 연도는 ${reviewData.firstYear}~${reviewData.lastYear}년입니다.`;
  return;
 }
 const result=reviewData.results[site?.name]?.[$('species').value]?.[year];
 if(!result?.ok){
  notice.textContent=result?.message??'선택한 관측지와 수종의 봄꽃 재점검 자료가 없습니다.';
  return;
 }
 notice.textContent=result.lead==='2월 말'?'해당 연도 2월 기온 반영':'과거 관측 기반 계획 참고값 · 해당 연도 기온 미반영';
 if(result.source==='집계표(임시)')notice.textContent+=' · 집계표 기반 개발용';
 $('reviewDate').textContent=result.pred_date;
 const interval=result.interval;
 $('reviewDetails').textContent=`${result.method} · ${result.n_years}년 기준 · ${interval.level*100}% 범위 ${interval.low_date} ~ ${interval.high_date}`;
 if(result.warnings.length){
  $('reviewWarnings').textContent=result.warnings.join(' ');
  $('reviewWarnings').classList.remove('hidden');
 }
}
function update(){const s=selectedSite();$('location').textContent=`${$('province').value} ${$('city').value}`;$('selection').textContent=s?`${s.name} · ${$('species').value} · ${$('phenomenon').value}`:'이 지역에는 연결된 정제 관측자료가 없습니다.';drawMap();analysis();plan();review();}
const headings={plan:['01 / PLAN','꽃이 머무는 시간에, 축제를 맞추다','과거 기록상 개화 구간으로 같은 일수의 축제 일정을 비교합니다.'],review:['02 / REVIEW','다가오는 봄, 일정을 다시 살펴보다','과거 개화 기록과 대상 연도의 개화 예상 시기를 함께 확인합니다.'],analysis:['03 / EXPLORE','해마다 달라지는 계절의 기록','연도별 관측일과 수종별 변동 폭을 관측 연수·품질과 함께 살펴봅니다.']};
document.querySelectorAll('nav button').forEach(button=>button.addEventListener('click',()=>{currentPage=button.dataset.page;document.body.className=currentPage;document.querySelectorAll('nav button').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});document.querySelectorAll('.page').forEach(s=>s.classList.toggle('hidden',!s.classList.contains(currentPage)));['step','title','description'].forEach((id,i)=>$(id).textContent=headings[currentPage][i]);}));
$('province').addEventListener('change',()=>loadCities());$('city').addEventListener('change',()=>{loadPlaces();focusRegion();});$('place').addEventListener('change',loadSpecies);$('phenomenon').addEventListener('change',loadSpecies);$('species').addEventListener('change',loadStages);['stage','includeSuspect','year'].forEach(id=>$(id).addEventListener('change',update));['start','end','searchStart','searchEnd'].forEach(id=>$(id).addEventListener('change',update));$('calculate').addEventListener('click',plan);$('mapReset').addEventListener('click',()=>{$('map').setAttribute('viewBox','0 0 660 660');});
$('alternatives').addEventListener('click',e=>{const button=e.target.closest('button[data-apply]');if(button){$('start').value=button.dataset.apply;$('end').value=button.dataset.end;update();}});
$('downloadCsv').addEventListener('click',()=>{const csvCell=s=>'"'+String(s).replaceAll('"','""')+'"';const content='\uFEFF'+[['연도','수목원','수종','관측단계','윤년기준환산DOY','나무수','이상치의심'],...selectedRows()].map(r=>r.map(csvCell).join(',')).join('\r\n');const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='Season Window_선택관측요약_내부용.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
options('province',[...new Set(DATA.regions.map(r=>r.province))],'경기도');options('phenomenon',['개화','단풍'],'개화');loadCities('포천시');$('stage').value='개화50%이상';update();
// Exposed pure calculations permit independent regression checks without synthetic UI results.
window.flowerApp={data:DATA,score,probability,selectedIntervals,validateDates,selectedRows,sd,md,projectDay};
