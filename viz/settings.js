const CODE = 'ArrowUp ArrowUp ArrowDown ArrowDown ArrowLeft ArrowRight ArrowLeft ArrowRight b a';
const PALETTE = ['#ffffff', '#f6d32d', '#ff7a45', '#4fd1ff', '#ff5fc8', '#8cff6b'];
const KEY = 'espoo.settings.2';
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };
const save = v => { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch {} };
const make = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };

const CONTROLS = {
  range: (t, set) => {
    const i = make('input', { type: 'range', min: t.r[0], max: t.r[1], step: t.r[2], oninput: () => set(+i.value) }), b = make('b');
    return [[i, b], v => { i.value = v; b.textContent = v; }];
  },
  toggle: (t, set) => {
    const b = make('button', { onclick: () => set(!b.classList.contains('on')) });
    return [[b], v => { b.classList.toggle('on', v); b.textContent = v ? 'on' : 'off'; }];
  },
  choice: (t, set) => {
    const bs = t.r.map(o => make('button', { textContent: o, onclick: () => set(o) }));
    return [[make('span', { className: 'opts' }, ...bs)], v => bs.forEach((b, k) => b.classList.toggle('on', t.r[k] === v))];
  },
  color: (t, set) => {
    const bs = PALETTE.map(c => make('button', { className: 'dot', onclick: () => set(c) }));
    bs.forEach((b, k) => { b.style.background = PALETTE[k]; });
    const i = make('input', { type: 'color', oninput: () => set(i.value) });
    return [[make('span', { className: 'opts' }, ...bs, i)], v => { i.value = v; bs.forEach((b, k) => b.classList.toggle('on', PALETTE[k] === v)); }];
  }
};

export function settings(table) {
  const val = { ...Object.fromEntries(table.map(t => [t.k, t.d])), ...load() }, show = {};
  const set = (t, v) => { val[t.k] = v; t.f(v); show[t.k](v); };
  const panel = make('aside', { id: 'cfg' }, make('button', { id: 'shut', textContent: 'x', onclick: () => panel.classList.remove('open') }), ...table.map(t => {
    const [nodes, upd] = CONTROLS[t.t](t, v => { set(t, v); save(val); });
    show[t.k] = upd;
    return make('div', { className: 'row' }, make('span', { textContent: t.l }), ...nodes);
  }), make('button', { id: 'reset', textContent: 'reset', onclick: () => { for (const t of table) set(t, t.d); save({}); } }));
  document.body.append(panel);
  for (const t of table) set(t, t.k in val ? val[t.k] : t.d);
  let keys = [];
  addEventListener('keydown', e => {
    keys = [...keys, e.key.length === 1 ? e.key.toLowerCase() : e.key].slice(-10);
    if (keys.join(' ') === CODE) { panel.classList.add('open'); keys = []; }
  });
}
