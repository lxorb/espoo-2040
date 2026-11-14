import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { City, HYPE, looks, patch, fadeDepth, borderMat } from './city.js';
import { settings } from './settings.js';
import { Near } from './near.js';
import { Detail } from './detail.js';
import { Net } from './traffic.js';
import { present } from './present.js';
import { Cars } from './cars.js';

const STATIONS = [['Kivenlahti', -1201, 1072], ['Espoonlahti', -205, 361], ['Soukka', 603, -544], ['Kaitaa', 1797, 312], ['Finnoo', 2981, 656],
  ['Matinkylä', 4518, 1343], ['Niittykumpu', 5884, 2487], ['Urheilupuisto', 6902, 2920], ['Tapiola', 8165, 2929], ['Aalto-yliopisto', 9295, 3944], ['Keilaniemi', 9590, 2952]];
const SKY = '#2b2e31', FLOW = { 1: 2, 2: 5, 3: 7.5, 4: 10, 5: 12 }, TILT = Math.tan(15 * Math.PI / 180);
const q = new URLSearchParams(location.search);
const opt = { gd: 1, heat: 0.3, flow: 0.07, tl: 0.47, dz: 1 }, flowOpt = {};
const state = { mode: q.get('mode') || 'cars', who: q.get('run'), f: +(q.get('year') || 0), play: false, run: null, ref: null, speed: 1, flow: 3, split: q.has('cut'), pair: [], shown: [], cut: +(q.get('cut') || 0.5) };

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: q.has('shot') });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 400000);
camera.zoom = +(q.get('zoom') || 3);
const controls = new MapControls(camera, renderer.domElement);
controls.enableDamping = false;
controls.minZoom = 0.03;
controls.maxZoom = 30;
controls.minPolarAngle = controls.maxPolarAngle = Math.acos(1 / Math.sqrt(3));
controls.zoomToCursor = true;
const [tx, ty] = (q.get('at') || HYPE.join(',')).split(',').map(Number);
controls.target.set(tx, 0, -ty);
camera.position.set(tx - 60000, 60000, -ty + 60000);
controls.update();

const hemi = new THREE.HemisphereLight('#e4ebf2', '#6d6a62', 1.2);
const sun = new THREE.DirectionalLight('#fff3e2', 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.6;
scene.add(hemi, sun, sun.target);

const city = new City(scene), near = new Near(scene, renderer, city), detail = new Detail(scene, city), cars = new Cars(scene), fineDepth = fadeDepth(looks.fine);
let nets = [];

const catalogue = fetch('data/catalogue.json').then(r => r.json()).catch(() => []);
fetch('data/net.json').then(r => r.json()).then(async n => {
  const cat = await catalogue;
  nets = [0, 1].map(() => Object.assign(new Net(scene, n, renderer, camera, cat), { opt: flowOpt }));
  for (const m of nets) state.ref && m.calib(state.ref);
  apply();
  runList.then(loadRun);
});
const hypeMats = [], fine = [];
let lod = 0, coarse = false, close = false;
function hull(pts) {
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = list => list.reduce((h, p) => { while (h.length > 1 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop(); h.push(p); return h; }, []);
  const lo = half(pts), hi = half([...pts].reverse());
  return lo.slice(0, -1).concat(hi.slice(0, -1));
}
function place(url, footprint) {
  new GLTFLoader().load(url, g => {
    const polys = [];
    g.scene.updateMatrixWorld(true);
    g.scene.traverse(o => {
      if (!o.isMesh) return;
      o.castShadow = o.receiveShadow = true;
      o.customDepthMaterial = fineDepth;
      for (const m of [o.material].flat()) { m.userData.base = m.color.clone(); hypeMats.push(patch(m, '', '', '', looks.one, looks.fine, !footprint)); }
      if (!footprint) return;
      const a = o.geometry.attributes.position, w = new THREE.Vector3(), pts = [];
      for (let i = 0; i < a.count; i++) { w.fromBufferAttribute(a, i).applyMatrix4(o.matrixWorld); pts.push([w.x, -w.z]); }
      polys.push(hull(pts));
    });
    scene.add(g.scene);
    fine.push(g.scene);
    g.scene.visible = lod < 1;
    if (polys.length) city.hide(polys);
    apply();
  }, undefined, () => {});
}
place('assets/hype.glb', true);
fetch('assets/district.json').then(r => r.json()).then(d => {
  city.hide((d.hide || []).map(f => f.reduce((p, v, i) => (i % 2 ? p[p.length - 1].push(v) : p.push([v]), p), [])));
  for (const m of d.models || []) place(`assets/${m}`, false);
}).catch(() => {});

const file = who => fetch(`data/runs/${who}.json`).then(r => r.json()), start = file('nothing');
const runs = {}, get = who => runs[who] ||= Promise.all([file(who), start]).then(([r, b]) => ({ ...r, y0: r.y0 - 1, years: [{ ...b.years[0], actions: [], works: [], new: [], open: [], events: [] }, ...r.years] }));
const runList = fetch('data/runs/index.json').then(r => r.json()).then(all => Promise.all(all.map(r => fetch(`data/runs/${r}.json`, { method: 'HEAD' }).then(x => x.ok && r, () => false)))).then(ok => {
  const list = ok.filter(Boolean);
  const box = document.getElementById('runs');
  for (const r of list) box.insertAdjacentHTML('beforeend', `<button data-v="${r}">${r.toUpperCase()}</button>`);
  box.insertAdjacentHTML('beforeend', '<button id="split" title="split screen (S)">split</button>');
  state.list = list;
  state.pair = list.slice(0, 2);
  state.split &&= state.pair.length === 2 && !state.who;
  state.who ||= list[1] || list[0];
  sync();
  return list;
});
get('nothing').then(r => { state.ref = r; for (const m of nets) m.calib(r); paint(); sync(); }).catch(() => {});
const blank = looks.clr[0].value, paint = () => {
  wake = 30;
  nets.forEach((m, k) => {
    m.on = !!state.shown[k];
    if (m.on) m.set(state.shown[k], state.f);
    m.show(state.mode);
    looks.clr[k].value = m.on && m.clr ? m.clr.tex : blank;
    if (m.on && m.clr) looks.cb[k].value.copy(m.clr.box); else looks.cb[k].value.set(0, 0, 0, 0);
  });
};
function loadRun() {
  const want = state.split ? state.pair : [state.who], key = want.join();
  Promise.all(want.map(get)).then(rs => {
    if (key !== (state.split ? state.pair : [state.who]).join()) return;
    state.shown = rs;
    sel = null;
    state.run = rs[0];
    year.max = Math.min(...rs.map(r => r.years.length)) - 1;
    state.f = Math.min(state.f, year.max);
    paint();
    sync();
  }).catch(() => want.forEach(w => { const b = document.querySelector(`#runs [data-v="${w}"]`); if (b) b.disabled = true; }));
}
const year = document.getElementById('year'), ylab = document.getElementById('ylab'), play = document.getElementById('play');
const cut = document.getElementById('cut'), sides = [...document.querySelectorAll('.side')];
sides.forEach((e, k) => e.onclick = () => { const o = state.list.find(r => !state.pair.includes(r)); if (o) { state.pair[k] = o; loadRun(); } });
const speed = document.getElementById('speed'), rate = () => state.mode === 'flow' ? 'flow' : 'speed';
function sync() {
  for (const b of document.querySelectorAll('#modes button')) b.classList.toggle('on', b.dataset.v === state.mode);
  for (const b of document.querySelectorAll('#runs button')) { b.classList.toggle('on', b.id === 'split' ? state.split : b.dataset.v === state.who); b.hidden = state.split && b.id !== 'split'; }
  play.classList.toggle('on', state.play);
  year.value = state.f;
  speed.textContent = `${state[rate()]}x`;
  ylab.textContent = (state.run ? state.run.y0 : 2025) + Math.floor(state.f + 1e-6);
  const split = state.split && state.shown.length === 2;
  cut.hidden = !split;
  sides.forEach((e, k) => { e.hidden = !split; e.textContent = state.pair[k]?.toUpperCase() ?? ''; });
}
function toggleSplit() {
  if (state.pair.length < 2) return;
  state.split = !state.split;
  loadRun();
  sync();
}
let drag = false;
cut.onpointerdown = e => { drag = true; cut.setPointerCapture(e.pointerId); };
cut.onpointermove = e => { if (drag) state.cut = Math.min(Math.max((e.clientX - (innerHeight / 2 - e.clientY) * TILT) / innerWidth, 0.02), 0.98); };
cut.onpointerup = cut.onpointercancel = () => { drag = false; };
cut.ondblclick = () => { state.cut = 0.5; };
const ground = (cx, cy) => { ray.setFromCamera({ x: cx / innerWidth * 2 - 1, y: 1 - cy / innerHeight * 2 }, camera); return ray.ray.intersectPlane(plane, hit) ? [hit.x, -hit.z] : null; };
function split() {
  const on = state.split && state.shown.length === 2, W = innerWidth, H = innerHeight, X = state.cut * W, pr = renderer.getPixelRatio();
  renderer.getDrawingBufferSize(looks.res.value);
  looks.ccl.value.set(0, 0, -1);
  if (!on) { if (nets[0]) { nets[0].uni.uClip.value.set(0, 0, 1); nets[0].cut = null; } return; }
  cut.style.left = `${X}px`;
  const top = X + (H / 2 - 30) * TILT;
  sides[0].style.right = `${W - top + 14}px`;
  sides[1].style.left = `${top + 14}px`;
  const a = ground(X + H / 2 * TILT, 0), b = ground(X - H / 2 * TILT, H), r = ground(X - 50, H / 2);
  const s = a && b && r ? Math.sign((r[0] - a[0]) * (b[1] - a[1]) - (r[1] - a[1]) * (b[0] - a[0])) : 0;
  nets.forEach((m, k) => {
    const g = k ? 1 : -1;
    m.uni.uClip.value.set(g / pr, -g * TILT / pr, g * (H / 2 * TILT - X));
    m.cut = s ? [...a, ...b, -g * s] : null;
  });
  looks.ccl.value.copy(nets[1].uni.uClip.value);
}
function apply() {
  wake = 30;
  looks.dim.value = { cars: 1, heat: opt.heat, flow: opt.flow }[state.mode];
  const c = new THREE.Color(SKY).multiplyScalar(state.mode === 'flow' ? 0 : looks.dim.value).getRGB({}, THREE.SRGBColorSpace);
  looks.sky.value.set(c.r, c.g, c.b);
  scene.background = new THREE.Color().setRGB(c.r, c.g, c.b, THREE.SRGBColorSpace);
  looks.sat.value = { cars: 1, heat: 0, flow: 0.25 }[state.mode];
  for (const m of hypeMats) m.userData.base && m.color.copy(m.userData.base).multiplyScalar(looks.dim.value);
  hemi.intensity = state.mode === 'flow' ? 0.5 : 1.2;
  for (const m of nets) m.show(state.mode);
  cars.show(state.mode);
  sync();
}
document.getElementById('modes').onclick = e => { if (e.target.dataset.v) { state.mode = e.target.dataset.v; apply(); } };
document.getElementById('runs').onclick = e => { if (e.target.id === 'split') toggleSplit(); else if (e.target.dataset.v) { state.who = e.target.dataset.v; loadRun(); sync(); } };
year.oninput = () => { state.f = +year.value; paint(); sync(); };
speed.onclick = () => { state[rate()] = state[rate()] % 5 + 1; sync(); };
play.onclick = () => { state.play = !state.play; if (state.play && state.f >= +year.max) state.f = 0; sync(); };
addEventListener('keydown', e => {
  if (e.key.toLowerCase() === 's') toggleSplit();
  if (e.key === ' ') play.onclick();
  if (e.key === 'o' || e.key === 'O') go(...VIEWS[e.key]);
  const m = { 1: 'cars', 2: 'heat', 3: 'flow' }[e.key];
  if (m) { state.mode = m; apply(); }
});

const UI = ['modes', 'right', 'time', 'sl', 'sr', 'card'].map(id => document.getElementById(id));
let tags = [], shown = [], espoo = null;
const HOME = -Math.PI / 4, turn = () => controls.getAzimuthalAngle();
const VIEWS = { hype: [...HYPE, 3, HOME], o: [2422, 7758, 0.052, HOME], O: [7437, 7120, 0.157, HOME], 0: [6379, 7698, 0.086, HOME] };
const go = (x, y, z1 = Math.max(camera.zoom, 2.5), r1 = turn()) => { fly = { t: 0, a: controls.target.clone(), b: new THREE.Vector3(x, 0, -y), z0: camera.zoom, z1, r0: turn(), r1 }; };
function label(text, x, y, h, cls) {
  const el = document.createElement('div');
  el.className = `lab ${cls}`;
  el.textContent = text;
  el[text === 'HYPE' ? 'onclick' : 'ondblclick'] = () => text === 'HYPE' ? go(...VIEWS.hype) : go(x, y);
  document.getElementById('labels').append(el);
  return { el, p: new THREE.Vector3(x, h, -y) };
}
const WORDS = { road: ['BUILD ROAD', 'NEW ROAD'], lanes: ['WIDEN ROAD', 'WIDER ROAD'], junction: ['REBUILD JUNCTION', 'NEW JUNCTION'] };
const words = (t, f) => f < t.a - 1e-6 || f >= t.b + 2.5 ? '' : (WORDS[t.what] || WORDS.road)[+(f >= t.b - 1e-6)];
let lit = null, sel = null;
const outlines = () => { wake = 30; for (const m of nets) m.outline(lit?.net === m ? lit : sel?.m === m ? sel : null); };
function hover(t) {
  lit?.el.classList.remove('hi');
  lit = t;
  t?.el.classList.add('hi');
  outlines();
}
function projectTags(put) {
  const live = nets.filter(m => m.on && m.tags);
  if (live.length !== shown.length || live.some((m, k) => m.tags !== shown[k])) {
    for (const t of tags) t.el.remove();
    hover(null);
    shown = live.map(m => m.tags);
    tags = live.flatMap(m => m.tags.map(t => {
      t.el ||= Object.assign(document.createElement('div'), { className: 'lab tag', onpointerenter: e => e.pointerType === 'mouse' && hover(t), onpointerleave: e => e.pointerType === 'mouse' && hover(null), onpointerup: e => e.pointerType !== 'mouse' && hover(lit === t ? null : t) });
      t.p ||= new THREE.Vector3(t.x, 20, -t.y);
      t.net = m;
      document.getElementById('labels').append(t.el);
      return t;
    }));
  }
  const w = innerWidth, h = innerHeight;
  for (const t of tags) {
    const text = words(t, state.f);
    if (t.el.textContent !== text) t.el.textContent = text;
    v.copy(t.p).project(camera);
    const x = (v.x + 1) / 2 * w, bw = text.length * 6.6 + 14;
    let y = (1 - v.y) / 2 * h - 8;
    t.on = !!text && t.net.keep(t.x, t.y) && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
    t.el.style.display = t.on ? '' : 'none';
    if (!t.on) continue;
    while (put.some(b => x - bw / 2 < b[2] + 4 && x + bw / 2 > b[0] - 4 && y - 20 < b[3] + 2 && y > b[1] - 2)) y -= 22;
    put.push([x - bw / 2, y - 20, x + bw / 2, y]);
    t.el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
    t.el.style.opacity = Math.min(Math.max(1 - (state.f - t.b - 2) / 0.5, 0), 1);
  }
  if (lit && !lit.on) hover(null);
}

const labels = [label('HYPE', ...HYPE, 40, 'big'), ...STATIONS.map(([n, x, y]) => label(n.toUpperCase(), x, y, 12, 'st'))];

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h);
  Object.assign(camera, { left: -w / 2, right: w / 2, top: h / 2, bottom: -h / 2 });
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

const ray = new THREE.Raycaster(), plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit = new THREE.Vector3(), v = new THREE.Vector3();
function view() {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    ray.setFromCamera({ x: a, y: b }, camera);
    if (!ray.ray.intersectPlane(plane, hit)) continue;
    x0 = Math.min(x0, hit.x); x1 = Math.max(x1, hit.x); y0 = Math.min(y0, -hit.z); y1 = Math.max(y1, -hit.z);
  }
  return [x0, y0, x1, y1];
}

let fly = null;
function flying(dt) {
  fly.t = Math.min(fly.t + dt / 1.2, 1);
  const e = fly.t * fly.t * (3 - 2 * fly.t), p = fly.a.clone().lerp(fly.b, e);
  const d = fly.r1 - fly.r0;
  controls.target.copy(p);
  camera.position.setFromSphericalCoords(60000 * Math.sqrt(3), controls.maxPolarAngle, fly.r0 + Math.atan2(Math.sin(d), Math.cos(d)) * e).add(p);
  camera.zoom = fly.z0 * (fly.z1 / fly.z0) ** e;
  camera.updateProjectionMatrix();
  if (fly.t === 1) fly = null;
}

const sunDir = new THREE.Vector3(-0.45, 0.8, 0.75).normalize(), sunX = new THREE.Vector3(0, 1, 0).cross(sunDir).normalize(), sunY = sunDir.clone().cross(sunX), snap = new THREE.Vector3();
const clock = new THREE.Clock();
let lastKey = '', wake = 0;
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (fly) flying(dt);
  coarse = state.mode !== 'cars' || camera.zoom < (coarse ? 1.0 : 0.8) * opt.dz;
  const lt = coarse ? 1 : 0;
  if (lt !== lod) {
    lod = Math.min(Math.max(lod + Math.sign(lt - lod) * dt / 0.25, 0), 1);
    looks.coarse.value = lod;
    looks.fine.value = 1 - lod;
    city.setLod(lod);
    for (const g of fine) g.visible = lod < 1;
  }
  close = state.mode !== 'flow' && camera.zoom > (close ? 0.8 : 1.0) * opt.dz;
  looks.near.value = Math.min(Math.max(looks.near.value + (close ? dt : -dt) / 0.25, 0), 1);
  looks.far.value = 1 - looks.near.value;
  looks.gtex.value = Math.min(Math.max(looks.gtex.value + (state.mode !== 'flow' && camera.zoom > (looks.gtex.value > 0 ? 0.65 : 0.7) ? dt : -dt) / 0.25, 0), 1);
  looks.detail.value = (state.mode !== 'flow') * Math.min(Math.max((camera.zoom - 0.45) / 0.45, 0), 1) * opt.gd * looks.ready.value;
  controls.update();
  const px = 1 / camera.zoom;
  const [x0, y0, x1, y1] = view();
  city.update(x0 - 200, y0 - 200, x1 + 200, y1 + 200);
  city.show(state.mode, camera.zoom);
  near.update(camera, [x0, y0, x1, y1], dt * state.speed, state.mode);
  detail.update(camera, lod);
  const span = Math.min(2 ** Math.ceil(Math.log2(Math.max(x1 - x0, y1 - y0) * 0.6)), 4096), texel = span / 2048;
  const t = snap.copy(controls.target);
  for (const a of [sunX, sunY]) { const s = t.dot(a); t.addScaledVector(a, Math.round(s / texel) * texel - s); }
  sun.position.copy(t).addScaledVector(sunDir, 5000);
  sun.target.position.copy(t);
  Object.assign(sun.shadow.camera, { left: -span, right: span, top: span, bottom: -span, near: 100, far: 12000 });
  sun.shadow.camera.updateProjectionMatrix();
  if (state.play && state.run) {
    state.f = Math.min(state.f + dt / opt.tl, +year.max);
    if (state.f >= +year.max) state.play = false;
    paint();
    sync();
  }
  split();
  if (state.mode === 'cars') cars.tick(dt * state.speed, nets.filter(m => m.on && m.links), [x0, y0, x1, y1], px, city.loaded);
  for (const m of nets) m.draw(state.mode, [x0, y0, x1, y1], dt * (state.mode === 'flow' ? FLOW[state.flow] : state.speed), px, city.loaded);
  if (city.tag && !espoo) labels.push(espoo = Object.assign(label('ESPOO', ...city.tag, 20, 'big'), { far: true }));
  const w = innerWidth, h = innerHeight, now = performance.now();
  const key = [camera.position.x, camera.position.z, camera.zoom, state.f, state.mode, state.cut, state.split, w, h, city.loaded.size, city.busy, near.ground.busy, detail.busy, looks.ready.value, lod].join();
  if (key !== lastKey || fly) wake = 30;
  lastKey = key;
  if (wake-- > 0 || state.mode !== 'heat' || looks.near.value > 0 || nets.some(m => now - m.xf < 600)) draw(w, h, []);
  frames++;
  requestAnimationFrame(frame);
}
function draw(w, h, put) {
  for (const l of labels) {
    v.copy(l.p).project(camera);
    l.on = Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 && (!l.el.classList.contains('st') || camera.zoom > 0.12) && (!l.far || camera.zoom < 0.35);
    l.el.style.display = l.on ? '' : 'none';
    if (!l.on) continue;
    const x = (v.x + 1) / 2 * w, y = (1 - v.y) / 2 * h;
    l.el.style.transform = `translate(${x}px, ${y - 8}px) translate(-50%, -100%)`;
    put.push([x - l.el.offsetWidth / 2, y - 8 - l.el.offsetHeight, x + l.el.offsetWidth / 2, y - 8]);
  }
  projectTags(put);
  const all = [...labels, ...tags];
  const ui = UI.map(e => e.getBoundingClientRect()), over = a => ui.some(b => a.left < b.right + 8 && a.right > b.left - 8 && a.top < b.bottom + 8 && a.bottom > b.top - 8);
  for (const l of all.filter(l => l.on && over(l.el.getBoundingClientRect()))) l.el.style.display = 'none';
  renderer.render(scene, camera);
  for (const m of nets) m.overlay(state.mode);
}
const el = (tag, cls, ...kids) => { const e = document.createElement(tag); if (cls) e.className = cls; e.append(...kids); return e; };
const card = document.getElementById('card');
let hp = null, tap = null;
function find(p) {
  const g = p && ground(...p), m = g && nets.find(m => m.on && m.links && m.keep(...g)), W = l => Math.max(l.lanes || 1, 1) * 3.3 + 1;
  let best = 12 * 1.8 / camera.zoom, k = -1;
  if (m) m.links.forEach((l, i) => {
    const b = m.box[i];
    if (g[0] < b[0] - best || g[0] > b[2] + best || g[1] < b[1] - best || g[1] > b[3] + best) return;
    const off = m.two[i] ? W(l) / 2 : 0;
    for (let s = 1; s < l.geom.length; s++) {
      const [ax, ay] = l.geom[s - 1], [bx, by] = l.geom[s], dx = bx - ax, dy = by - ay, d = Math.hypot(dx, dy) || 1, ox = dy / d * off, oy = -dx / d * off;
      const t = Math.min(Math.max(((g[0] - ax - ox) * dx + (g[1] - ay - oy) * dy) / (d * d), 0), 1), e = Math.hypot(ax + ox + dx * t - g[0], ay + oy + dy * t - g[1]);
      if (e < best) { best = e; k = i; }
    }
  });
  return k < 0 ? null : { m, k, hl: [m.two[k] !== undefined ? k : { geom: m.links[k].geom, w: W(m.links[k]) }] };
}
function pick() {
  const r = sel || find(hp);
  card.hidden = !r;
  if (!r) return;
  const { m, k } = r, l = m.links[k], pc = v => `${Math.round(v * 100)}%`;
  card.replaceChildren(el('b', '', l.name || l.ref || l.cls), ...[['flow', `${Math.round(m.flow[k])} veh/h`], ['queue', pc(m.loss[k])], ['condition', pc(m.cond[k])],
    ...m.cap[k] < 1 ? [['works', `${pc(m.cap[k])} capacity`]] : []].map(([a, b]) => el('div', '', el('span', '', a), el('span', '', b))));
}
const cv = renderer.domElement;
cv.addEventListener('pointermove', e => { if (e.pointerType === 'mouse') hp = [e.clientX, e.clientY]; });
cv.addEventListener('pointerleave', () => { hp = null; });
cv.addEventListener('pointerdown', e => { tap = [e.clientX, e.clientY]; });
cv.addEventListener('pointerup', e => { if (tap && Math.hypot(e.clientX - tap[0], e.clientY - tap[1]) < 6) { sel = find(tap); outlines(); pick(); } });
setInterval(() => { if (hp || sel || !card.hidden) pick(); }, 60);

const fm = document.getElementById('fm'), rules = fetch('data/rules.json').then(r => r.json()).catch(() => ({}));
const span = (a, u = '') => { const lo = Math.min(...a), hi = Math.max(...a); return a.length ? `${lo === hi ? lo : `${lo} to ${hi}`}${u}` : '?'; };
const KINDS = [[p => /-access$/.test(p.id), 'Area access upgrade', 'areas', 'near one growth area'], [p => p.kind === 'road', 'New road', 'roads'],
  [p => p.kind === 'lanes', 'Widen road', 'roads', 'on one road'], [p => p.kind === 'junction', 'Rebuild junction', 'junctions', 'at one junction']];
function table(head, rows, cls = '') {
  const t = el('table', '', ...head ? [el('tr', '', ...head.map(h => el('th', '', h)))] : [], ...rows.map(([c, ...r]) => el('tr', c, ...r.map(v => el('td', '', v)))));
  return el('div', `t ${cls}`, t);
}
async function formal() {
  const [cat, R] = await Promise.all([catalogue, rules]), q = v => v ?? '?', loss = ps => ps.flatMap(p => (p.works || []).map(w => Math.round((1 - w.cap) * 100)));
  const groups = [], left = new Set(cat);
  for (const [f, name, noun, where] of KINDS) { const ps = cat.filter(p => left.has(p) && f(p)); ps.forEach(p => left.delete(p)); if (ps.length) groups.push([name, noun, where, ps]); }
  for (const p of left) groups.push([p.kind, 'projects', '', [p]]);
  const rows = groups.map(([name, noun, where, ps]) => {
    const caps = ps.flatMap(p => (p.mod || []).map(m => m.cap).filter(Boolean)), w = loss(ps);
    const what = where ? (caps.every(c => c === 2) ? 'Doubles capacity' : `${span(caps)} times the capacity`) + ` ${where}` : `Adds a new ${span(ps.flatMap(p => (p.new || []).map(n => n.lanes)))} lane road`;
    return ['i', name, `${ps.length} ${noun}`, `${span(ps.map(p => p.cost), ' MEUR')}, ${span(ps.map(p => p.years), ' years')}`, what + (w.length ? `; roads lose ${span(w, '%')} capacity while it is built` : '')];
  });
  const M = R.maint || [];
  const parts = [['Action Space', table(['What', 'Options', 'Cost', 'Effect'], [['g', 'Start a road project', `${cat.length} projects, up to ${q(R.builds)} per year`, '', ''], ...rows,
      ['', 'Set maintenance level', `${M.length || '?'} levels`, `${M.join(' / ') || '?'} MEUR per year`, 'Repairs worn roads; worn roads are slower'],
      ['', 'Save money', 'Spend nothing', '0', 'Unspent money carries over to the next year']])],
    ['Observation Space', table(null, [['', 'Areas', 'Population and jobs of every area, year by year'], ['', 'Projects', 'Cost, build time and share of delay on its roads'],
      ['', 'Roads', 'The most congested roads'], ['', 'Forecasts', `Up to ${q(R.forecasts)} forecasts, ${q(R.forecast_years)} years ahead each`]], 'n')],
    ['Limits', table(null, [['', 'Budget', `${q(R.income)} MEUR per year`], ['', 'Projects', `At most ${q(R.builds)} new projects per year`], ['', 'Building', `Roads under works lose ${span(loss(cat), '%')} capacity`],
      ['', 'Time', `${q(R.years)} years of decisions, then ${q(R.runout)} more years are scored`],
      ['', 'Hidden growth', `${R.booms ? span(R.booms) : '?'} of ${q(R.areas)} areas grow ${R.boom_growth ? span(R.boom_growth) : '?'} times in ${q(R.years)} years, without warning`]], 'n'),
      el('p', 's', `Score: total hours lost in traffic over ${R.years && R.runout ? R.years + R.runout : '?'} years. Doing nothing scores 0, the best known plan scores 1.`)]];
  const tabs = parts.map(([name], k) => Object.assign(el('button', '', name), { onclick: () => show(k) })), body = el('div', '');
  const show = k => { tabs.forEach((b, j) => b.classList.toggle('on', j === k)); body.replaceChildren(...parts[k].slice(1)); };
  show(0);
  document.getElementById('fmc').replaceChildren(el('p', '', `Each year, for ${q(R.years)} years, the planner makes one decision.`), el('div', 'seg tabs', ...tabs), body);
  fm.hidden = false;
}
document.getElementById('formal').onclick = formal;
document.getElementById('fx').onclick = () => { fm.hidden = true; };
fm.onclick = e => { if (e.target === fm) fm.hidden = true; };
addEventListener('keydown', e => { if (e.key === 'Escape') fm.hidden = true; });

present({
  go: k => go(...VIEWS[k]),
  set: o => { Object.assign(state, o); if ('who' in o || 'split' in o || 'pair' in o) loadRun(); if ('f' in o) paint(); apply(); },
  slide: to => { const a = state.cut, t0 = performance.now(), step = () => { const t = Math.min((performance.now() - t0) / 1500, 1); state.cut = a + (to - a) * t * t * (3 - 2 * t); if (t < 1) requestAnimationFrame(step); }; step(); }
});
apply();
const fps = Object.assign(document.createElement('div'), { id: 'fps' }), rgb = v => [1, 3, 5].map(i => parseInt(v.slice(i, i + 2), 16) / 255);
document.body.append(fps);
let frames = 0, since = 0;
setInterval(() => { fps.textContent = `${Math.round(frames / ((performance.now() - since) / 1000))} fps`; frames = 0; since = performance.now(); }, 500);
const knob = (k, l, t, r, d, f) => ({ k, l, t, r, d, f });
settings([
  knob('mark', 'marker color', 'color', 0, '#ffffff', v => { looks.mark.value.set(...rgb(v)); document.body.style.setProperty('--mark', v); }),
  knob('out', 'outside dim', 'range', [0, 1, 0.05], 0.1, v => { looks.out.value = v; apply(); }),
  knob('border', 'border', 'toggle', 0, true, v => { borderMat.visible = v; }),
  knob('labels', 'labels', 'toggle', 0, true, v => { document.getElementById('labels').hidden = !v; }),
  knob('shadows', 'shadows', 'toggle', 0, true, v => { sun.castShadow = v; }),
  knob('res', 'resolution', 'choice', [0.5, 1, 'native'], 'native', v => { renderer.setPixelRatio(v === 'native' ? Math.min(devicePixelRatio, 2) : v); resize(); }),
  knob('gd', 'ground detail', 'toggle', 0, true, v => { opt.gd = +v; }),
  knob('sat', 'satellite buildings', 'toggle', 0, false, v => { detail.use(v); }),
  knob('dz', 'detail zoom', 'range', [0.5, 4, 0.1], 1, v => { opt.dz = v; }),
  knob('cars', 'max cars', 'range', [0, 20000, 500], 20000, v => { cars.max = v; }),
  knob('cz', 'car spawn zoom', 'range', [0.3, 5, 0.1], 1, v => { cars.spawn = v; }),
  knob('walk', 'pedestrians', 'range', [0, 1, 0.1], 1, v => { near.walkers.density = v; near.key = ''; }),
  knob('trees', 'trees', 'range', [0, 1, 0.1], 1, v => { looks.trees.value = v; near.key = ''; }),
  knob('hr', 'heat radius', 'range', [0.25, 3, 0.05], 1, v => { flowOpt.radius = v; }),
  knob('hi', 'heat intensity', 'range', [0.25, 4, 0.05], 1, v => { flowOpt.heat = v; }),
  knob('fp', 'flow particles', 'range', [0.1, 1, 0.05], 1, v => { flowOpt.particles = v; }),
  knob('ft', 'flow trail', 'range', [0.25, 4, 0.05], 1, v => { flowOpt.trail = v; }),
  knob('fc', 'flow color', 'choice', ['relative', 'absolute'], 'relative', v => { flowOpt.abs = v === 'absolute'; }),
  knob('cb', 'congestion against', 'choice', ['2025', 'free flow'], '2025', v => { flowOpt.free = v === 'free flow'; for (const m of nets) m.lastF = NaN; paint(); }),
  knob('hb', 'heat background', 'range', [0.05, 1, 0.05], 0.2, v => { opt.heat = v; apply(); }),
  knob('fb', 'flow background', 'range', [0.02, 1, 0.01], 0.2, v => { opt.flow = v; apply(); }),
  knob('sy', 'seconds per year', 'range', [0.1, 5, 0.01], 0.47, v => { opt.tl = v; }),
  knob('fps', 'fps', 'toggle', 0, false, v => { fps.hidden = !v; })
]);
frame();
