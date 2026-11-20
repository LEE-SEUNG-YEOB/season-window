// Season Window 공통 동작
document.documentElement.classList.add('js');
window.SW = {
  reduce: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  el(t, cls, html){ const e = document.createElement(t); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; },
  md(d){ const x = new Date(Date.UTC(2025,0,1) + (Math.round(d)-1)*864e5); return `${x.getUTCMonth()+1}월 ${x.getUTCDate()}일`; },
  mdShort(d){ const x = new Date(Date.UTC(2025,0,1) + (Math.round(d)-1)*864e5); return `${x.getUTCMonth()+1}/${x.getUTCDate()}`; },
  // 화면에 처음 들어올 때 한 번만 등장 (.r 덩어리, 안쪽 .m 표시)
  reveal(root = document){
    const items = [...root.querySelectorAll('.r:not(.in)')];
    if (!('IntersectionObserver' in window)) { items.forEach(x => x.classList.add('in')); return; }
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); e.target.dispatchEvent(new CustomEvent('revealed')); }
    }), { rootMargin: '0px 0px -12% 0px', threshold: .01 });
    items.forEach(x => io.observe(x));
  },
  // 교체되는 글자: 4px 위로 들어오는 짧은 교체
  swap(node, text){
    if (node.textContent === text) return;
    node.textContent = text;
    if (SW.reduce) return;
    node.style.animation = 'none'; void node.offsetWidth; node.style.animation = 'swap-in 200ms ease-out both';
  },
  dotChart(id, rows, opt){
    const { el } = SW, box = document.getElementById(id), pct = v => ((v - opt.lo) / (opt.hi - opt.lo) * 100).toFixed(2) + '%';
    rows.forEach(r => {
      const row = el('div','row'); row.append(el('div','lab', r.name));
      if (opt.date) row.append(el('div','date', opt.date(r)));
      const t = el('div','t');
      opt.grid.forEach(g => { const i = el('i','grid'); i.style.left = pct(g); t.append(i); });
      if (r.lo != null){ const ci = el('i','ci m gx' + (opt.gray ? ' gray' : '')); ci.style.left = pct(r.lo); ci.style.width = `calc(${pct(r.hi)} - ${pct(r.lo)})`; t.append(ci); }
      if (r.ring != null){
        const ln = el('i','link-line m gx'); t.append(ln);
        const rg = el('i','pt ring m pop'); rg.style.left = pct(r.ring); t.append(rg);
        r._line = ln;
      }
      const p = el('i','pt m pop' + (opt.gray ? ' gray' : '')); p.style.left = pct(r.v); t.append(p);
      t.title = opt.tip(r);
      const val = el('div','val', opt.fmt(r.v));
      row.append(t, val); box.append(row);
      r._pt = p; r._val = val; r._pct = pct; r._t = t; r._base = r.v;
      if (r._line){ const a = Math.min(r.ring, r.v), b = Math.max(r.ring, r.v); r._line.style.left = pct(a); r._line.style.width = `calc(${pct(b)} - ${pct(a)})`; }
    });
    const ax = el('div','axis-row');
    const sync = () => { ax.style.gridTemplateColumns = getComputedStyle(box.querySelector('.row')).gridTemplateColumns; };
    for (let i = 0; i < (opt.date ? 2 : 1); i++) ax.append(el('div'));
    const tk = el('div','ticks'); opt.grid.forEach(g => { const s = el('span', (opt.minor || []).includes(g) ? 'minor' : null, opt.tick(g)); s.style.left = pct(g); tk.append(s); });
    ax.append(tk, el('div')); box.append(ax); sync(); window.addEventListener('resize', sync);
    return rows;
  }
};
// 페이지가 스크립트를 다 그린 뒤 등장 감시 시작
window.addEventListener('DOMContentLoaded', () => requestAnimationFrame(() => SW.reveal()));
