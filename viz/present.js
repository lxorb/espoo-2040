import { globe } from './globe.js';

const el = (tag, props, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const CARDS = ['Build road', 'Widen road', 'Set maintenance level', 'Area access upgrade', 'Rebuild junction'];
const hide = id => document.getElementById(id)?.remove();
const show = (id, ...kids) => { hide(id); return document.body.appendChild(el('div', { id, className: 'pr' }, ...kids)); };

export function present(v) {
  const steps = [
    () => { v.set({ mode: 'cars', who: 'gpt-6.1', split: false, play: false, f: 0 }); v.go('o'); },
    () => show('pc', ...CARDS.map((t, k) => el('div', { textContent: t, style: `animation-delay: ${k * 0.12}s` }))),
    () => { hide('pc'); v.go('hype'); },
    () => v.set({ mode: 'flow' }),
    () => v.go('o'),
    () => show('pf', el('img', { src: 'assets/formulas.png' })),
    () => {
      hide('pf');
      v.set({ mode: 'cars', f: 1 });
      v.go('o');
      const r = document.getElementById('ylab').getBoundingClientRect();
      Object.assign(show('pa').style, { left: `${r.left + r.width / 2}px`, bottom: `${innerHeight - r.top + 32}px` });
    },
    () => { v.set({ mode: 'heat' }); show('pl', el('span', { textContent: 'Bad traffic' }), el('i'), el('span', { textContent: 'Good traffic' })); },
    () => { hide('pa'); v.set({ play: true, stop: 10, onstop: () => { v.go(3); v.set({ mode: 'cars' }); } }); },
    () => v.set({ mode: 'flow' }),
    () => v.set({ f: 11 }),
    () => { v.go('o'); v.set({ mode: 'cars', play: true }); },
    () => { v.go('O'); v.set({ mode: 'heat' }); },
    () => { hide('pl'); v.go(2); v.set({ mode: 'cars' }); },
    () => v.set({ mode: 'flow' }),
    () => { v.go(0); v.set({ mode: 'heat', split: true, pair: ['rl', 'gpt-6.1'], cut: 0.2635 }); },
    () => { v.go(1); v.slide(0.7955); },
    () => {
      const col = el('div', {}, ...Array.from({ length: pages }, (_, k) => el('img', { src: `assets/paper/${k + 1}.jpg` }))), box = show('pp', col), t0 = performance.now() + 1000;
      let y = 0, speed = 0, last = t0;
      const step = now => {
        if (!box.isConnected) return;
        const top = 0.97 * innerHeight / v.paper(), end = col.offsetHeight - innerHeight, dt = Math.min(Math.max(now - last, 0), 50) / 1000;
        last = Math.max(now, t0);
        speed = Math.min(speed + top * dt / 0.8, top);
        y = Math.min(y + speed * dt, end);
        col.style.transform = `translateY(${-y}px)`;
        if (y < end) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    },
    () => { hide('pp'); v.go('far'); g.show(); setTimeout(() => n > 0 && v.hold(true), 950); }
  ];
  let n = -1, taps = [], g, ready = false, skip = false, pages = 1;
  const mode = on => { n = on ? 0 : -1; ['pf', 'pa', 'pc', 'pp', 'pl', 'pw', 'pi'].forEach(hide); g?.hide(); v.hold(false); v.free(); ready = skip = false; if (on) { g ||= globe(); show('pi', 'p'); show('pw', 'Preloading'); v.warm(() => skip || n < 0).then(() => { hide('pw'); ready = n >= 0; }); new Image().src = 'assets/formulas.png'; fetch('assets/paper/pages.json').then(r => r.json()).then(j => { pages = j.n; for (let k = 1; k <= pages; k++) new Image().src = `assets/paper/${k}.jpg`; }); } };
  addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' && n >= 0 && ready) n < steps.length ? steps[n++]() : mode(false);
    if (e.key !== 'p') return;
    if (n >= 0 && !ready) skip = true;
    taps = [...taps, performance.now()].slice(-3);
    if (taps.length < 3 || taps[2] - taps[0] > 900) return;
    taps = [];
    mode(n < 0);
  });
}
