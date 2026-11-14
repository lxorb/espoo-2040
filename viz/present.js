const el = (tag, props, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const CARDS = ['Build road', 'Widen road', 'Set maintenance level', 'Area access upgrade', 'Rebuild junction'];
const hide = id => document.getElementById(id)?.remove();
const show = (id, ...kids) => { hide(id); return document.body.appendChild(el('div', { id, className: 'pr' }, ...kids)); };

export function present(v) {
  const steps = [
    () => { v.set({ mode: 'cars', who: 'gpt-6.1', split: false, play: false, f: 0 }); v.go('o'); },
    () => v.go('hype'),
    () => v.set({ mode: 'flow' }),
    () => show('pf', el('img', { src: 'assets/formulas.png' })),
    () => {
      hide('pf');
      v.set({ mode: 'cars', f: 1 });
      const r = document.getElementById('ylab').getBoundingClientRect();
      Object.assign(show('pa').style, { left: `${r.left + r.width / 2}px`, bottom: `${innerHeight - r.top + 32}px` });
    },
    () => { hide('pa'); show('pc', ...CARDS.map((t, k) => el('div', { textContent: t, style: `animation-delay: ${k * 0.12}s` }))); },
    () => { hide('pc'); v.go('o'); },
    () => v.set({ mode: 'heat' }),
    () => v.set({ play: true }),
    () => v.go('O'),
    () => { v.go(0); v.set({ split: true, pair: ['rl', 'gpt-6.1'], cut: 0.2635 }); },
    () => { v.go(1); v.slide(0.7955); },
    () => v.set({ pair: ['gemma', 'gpt-6.1'] }),
    () => show('pp', el('img', { src: 'assets/paper.jpg' }))
  ];
  let n = -1, taps = [];
  const mode = on => { n = on ? 0 : -1; ['pf', 'pa', 'pc', 'pp'].forEach(hide); if (on) { v.warm(); for (const f of ['formulas.png', 'paper.jpg']) new Image().src = `assets/${f}`; } };
  addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' && n >= 0) n < steps.length ? steps[n++]() : mode(false);
    if (e.key !== 'p') return;
    taps = [...taps, performance.now()].slice(-3);
    if (taps.length < 3 || taps[2] - taps[0] > 900) return;
    taps = [];
    mode(n < 0);
  });
}
