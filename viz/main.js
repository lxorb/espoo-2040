import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { City, HYPE, looks, patch, fadeDepth, borderMat } from './city.js';
import { settings } from './settings.js';
import { Near } from './near.js';
import { Detail } from './detail.js';
import { Net } from './traffic.js';
import { Cars } from './cars.js';

const STATIONS = [['Kivenlahti', -1201, 1072], ['Espoonlahti', -205, 361], ['Soukka', 603, -544], ['Kaitaa', 1797, 312], ['Finnoo', 2981, 656],
  ['Matinkylä', 4518, 1343], ['Niittykumpu', 5884, 2487], ['Urheilupuisto', 6902, 2920], ['Tapiola', 8165, 2929], ['Aalto-yliopisto', 9295, 3944], ['Keilaniemi', 9590, 2952]];
const RUNS = ['agent', 'rl'];
const SKY = '#2b2e31', FLOW = { 1: 2, 2: 5, 5: 12 }, TILT = Math.tan(15 * Math.PI / 180);
const q = new URLSearchParams(location.search);
const opt = { gd: 1, heat: 0.3, flow: 0.07, tl: 1.4, dz: 1, cars: 20000, fps: false }, flowOpt = { particles: 1, trail: 1, radius: 1, heat: 1, sources: false, abs: false };
const state = { mode: q.get('mode') || 'cars', who: q.get('run'), f: +(q.get('year') || 0), play: false, run: null, ref: null, speed: 1, flow: 1, split: false, pair: [], shown: [], cut: 0.5 };

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

const city = new City(scene), near = new Near(scene, renderer, city), detail = new Detail(scene, city), fineDepth = fadeDepth(looks.fine);
let nets = [], cars;

const catalogue = fetch('data/catalogue.json').then(r => r.json()).catch(() => []);
fetch('data/net.json').then(r => r.json()).then(async n => {
  const cat = await catalogue;
  nets = [0, 1].map(() => Object.assign(new Net(scene, n, renderer, camera, cat), { opt: flowOpt }));
  for (const m of nets) state.ref && m.calib(state.ref);
  apply();
  runList.then(loadRun);
});
cars = new Cars(scene);
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
      for (const m of [o.material].flat()) { m.userData.base = m.color.clone(); hypeMats.push(patch(m, '', '', '', looks.one, looks.fine)); }
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

const runs = {}, get = who => runs[who] ||= fetch(`data/runs/${who}.json`).then(r => r.json());
const runList = fetch('data/runs/index.json').then(r => r.json()).catch(() => RUNS).then(list => {
  const box = document.getElementById('runs');
  for (const r of list) box.insertAdjacentHTML('beforeend', `<button data-v="${r}">${r.toUpperCase()}</button>`);
  box.insertAdjacentHTML('beforeend', '<button id="split" title="split screen (S)">split</button>');
  state.pair = list.slice(0, 2);
  state.split &&= state.pair.length === 2 && !state.who;
  state.who ||= list[0];
  sync();
});
get('nothing').then(r => { state.ref = r; for (const m of nets) m.calib(r); paint(); sync(); }).catch(() => {});
const paint = () => nets.forEach((m, k) => {
  m.on = !!state.shown[k];
  if (m.on) m.set(state.shown[k], state.f);
  m.show(state.mode);
});
function loadRun() {
  const want = state.split ? state.pair : [state.who], key = want.join();
  Promise.all(want.map(get)).then(rs => {
    if (key !== (state.split ? state.pair : [state.who]).join()) return;
    state.shown = rs;
    state.run = rs[0];
    year.max = Math.min(...rs.map(r => r.years.length)) - 1;
    state.f = Math.min(state.f, year.max);
    paint();
    sync();
  }).catch(() => want.forEach(w => { const b = document.querySelector(`#runs [data-v=${w}]`); if (b) b.disabled = true; }));
}
const delay = (run, f) => { const ys = run.years, a = Math.min(Math.floor(f), ys.length - 1), b = Math.min(a + 1, ys.length - 1), t = f - a; return ys[a].delay * (1 - t) + ys[b].delay * t; };
const year = document.getElementById('year'), ylab = document.getElementById('ylab'), play = document.getElementById('play');
const cut = document.getElementById('cut'), sides = [...document.querySelectorAll('.side')];
const speed = document.getElementById('speed'), rate = () => state.mode === 'flow' ? 'flow' : 'speed';
function sync() {
  for (const b of document.querySelectorAll('#modes button')) b.classList.toggle('on', b.dataset.v === state.mode);
  for (const b of document.querySelectorAll('#runs button')) { b.classList.toggle('on', b.id === 'split' ? state.split : b.dataset.v === state.who); b.hidden = state.split && b.id !== 'split'; }
  play.classList.toggle('on', state.play);
  year.value = state.f;
  speed.textContent = `${state[rate()]}x`;
  ylab.textContent = (state.run ? state.run.y0 : 2026) + Math.floor(state.f + 1e-6);
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
}
function apply() {
  looks.dim.value = { cars: 1, heat: opt.heat, flow: opt.flow }[state.mode];
  const c = new THREE.Color(SKY).multiplyScalar(state.mode === 'flow' ? 0 : looks.dim.value).getRGB({}, THREE.SRGBColorSpace);
  looks.sky.value.set(c.r, c.g, c.b);
  scene.background = new THREE.Color().setRGB(c.r, c.g, c.b, THREE.SRGBColorSpace);
  looks.sat.value = { cars: 1, heat: 0, flow: 0.25 }[state.mode];
  for (const m of hypeMats) m.userData.base && m.color.copy(m.userData.base).multiplyScalar(looks.dim.value);
  hemi.intensity = state.mode === 'flow' ? 0.5 : 1.2;
  for (const m of nets) m.show(state.mode);
  cars && cars.show(state.mode);
  sync();
}
document.getElementById('modes').onclick = e => { if (e.target.dataset.v) { state.mode = e.target.dataset.v; apply(); } };
document.getElementById('runs').onclick = e => { if (e.target.id === 'split') toggleSplit(); else if (e.target.dataset.v) { state.who = e.target.dataset.v; loadRun(); sync(); } };
year.oninput = () => { state.f = +year.value; paint(); sync(); };
speed.onclick = () => { state[rate()] = { 1: 2, 2: 5, 5: 1 }[state[rate()]]; sync(); };
play.onclick = () => { state.play = !state.play; if (state.play && state.f >= +year.max) state.f = 0; sync(); };
addEventListener('keydown', e => {
  if (e.key.toLowerCase() === 's') toggleSplit();
  if (e.key === ' ') play.onclick();
  const m = { 1: 'cars', 2: 'heat', 3: 'flow' }[e.key];
  if (m) { state.mode = m; apply(); }
});

const UI = ['modes', 'runs', 'time', 'sl', 'sr'].map(id => document.getElementById(id));
let tags = [], shown = [], hold = 0, espoo = null;
function tag(text, x, y, cls) {
  const el = document.createElement('div');
  el.className = `lab ${cls}`;
  el.textContent = text;
  document.getElementById('labels').append(el);
  return { el, p: new THREE.Vector3(x, 20, -y) };
}
const WORDS = { road: ['BUILD ROAD', 'NEW ROAD'], lanes: ['WIDEN ROAD', 'WIDER ROAD'], junction: ['REBUILD JUNCTION', 'NEW JUNCTION'], rail: ['BUILD RAIL', 'NEW RAIL'],
  bus: ['ADD BUS LINE', 'NEW BUS LINE'], buslane: ['ADD BUS LANE', 'NEW BUS LANE'], toll: ['ADD TOLL', 'TOLL', 1], parking: ['CHARGE PARKING', 'PAID PARKING', 1], pnr: ['BUILD PARK & RIDE', 'PARK & RIDE', 1] };
function words(t, f, y0) {
  if (f < t.a - 1e-6) return '';
  const [doing, done, stays] = WORDS[t.what] || WORDS.road;
  if (f < t.b - 1e-6) return t.build ? doing : '';
  return stays || f < t.b + 2.5 ? done : '';
}
function projectTags(put) {
  const live = nets.filter(m => m.on && m.tags);
  if (live.length !== shown.length || live.some((m, k) => m.tags !== shown[k])) {
    for (const t of tags) t.el.remove();
    shown = live.map(m => m.tags);
    tags = live.flatMap(m => m.tags.map(t => {
      t.el ||= Object.assign(document.createElement('div'), { className: 'lab tag' });
      t.p ||= new THREE.Vector3(t.x, 20, -t.y);
      t.net = m;
      document.getElementById('labels').append(t.el);
      return t;
    }));
  }
  const y0 = state.run?.y0 ?? 2026, cap = camera.zoom < 0.2 ? 3 : camera.zoom < 0.5 ? 6 : 99, w = innerWidth, h = innerHeight;
  let n = 0;
  for (const t of tags) {
    const text = words(t, state.f, y0);
    if (t.el.textContent !== text) t.el.textContent = text;
    v.copy(t.p).project(camera);
    const x = (v.x + 1) / 2 * w, y = (1 - v.y) / 2 * h - 8, bw = text.length * 6.6 + 14, r = [x - bw / 2, y - 20, x + bw / 2, y];
    t.on = !!text && t.net.keep(t.x, t.y) && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 && n < cap && !put.some(b => r[0] < b[2] + 4 && r[2] > b[0] - 4 && r[1] < b[3] + 2 && r[3] > b[1] - 2);
    t.el.style.display = t.on ? '' : 'none';
    if (!t.on) continue;
    n++;
    put.push(r);
    t.el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
    t.el.style.opacity = t.kind === 'road' ? Math.min(Math.max(1 - (state.f - t.b - 2) / 0.5, 0), 1) : 1;
  }
}

let labels = [['HYPE', ...HYPE, 40, 'big'], ...STATIONS.map(([n, x, y]) => [n.toUpperCase(), x, y, 12, 'st'])].map(([text, x, y, h, cls]) => {
  const el = document.createElement('div');
  el.className = `lab ${cls}`;
  el.textContent = text;
  el.ondblclick = () => { fly = { t: 0, a: controls.target.clone(), b: new THREE.Vector3(x, 0, -y), z0: camera.zoom, z1: Math.max(camera.zoom, 2.5) }; };
  document.getElementById('labels').append(el);
  return { el, p: new THREE.Vector3(x, h, -y) };
});

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
  camera.position.add(p.clone().sub(controls.target));
  controls.target.copy(p);
  camera.zoom = fly.z0 + (fly.z1 - fly.z0) * e;
  camera.updateProjectionMatrix();
  if (fly.t === 1) fly = null;
}

const sunDir = new THREE.Vector3(-0.45, 0.8, 0.75).normalize(), sunX = new THREE.Vector3(0, 1, 0).cross(sunDir).normalize(), sunY = sunDir.clone().cross(sunX), snap = new THREE.Vector3();
const clock = new THREE.Clock();
let last = '';
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
  looks.gtex.value = Math.min(Math.max(looks.gtex.value + (state.mode !== 'flow' && camera.zoom > (looks.gtex.value > 0 ? 0.45 : 0.5) ? dt : -dt) / 0.25, 0), 1);
  looks.detail.value = (state.mode !== 'flow') * Math.min(Math.max((camera.zoom - 0.25) / 0.25, 0), 1) * opt.gd * looks.ready.value;
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
  if (state.play && state.run && (hold -= dt) < 0) {
    let f = Math.min(state.f + dt / opt.tl, +year.max);
    if (Math.floor(f) > Math.floor(state.f + 1e-6) && nets.some(m => m.on && m.opens?.has(Math.floor(f)))) { f = Math.floor(f); hold = 1.3; }
    state.f = f;
    if (state.f >= +year.max) state.play = false;
    paint();
    sync();
  }
  if (cars) cars.max = opt.cars;
  split();
  if (cars && state.mode === 'cars') cars.tick(dt * state.speed, nets.filter(m => m.on && m.links), [x0, y0, x1, y1], px, city.loaded);
  for (const m of nets) m.draw(state.mode, [x0, y0, x1, y1], dt * (state.mode === 'flow' ? FLOW[state.flow] : state.speed), px, city.loaded);
  if (city.tag && !espoo) labels.push(espoo = Object.assign(tag('ESPOO', ...city.tag, 'big'), { far: true }));
  const w = innerWidth, h = innerHeight, put = [];
  labels = labels.filter(l => l.el.isConnected);
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
  const s = `${city.loaded.size}`;
  if (s !== last) { last = s; document.body.dataset.tiles = s; }
  frames++;
  requestAnimationFrame(frame);
}
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
  knob('dz', 'detail zoom', 'range', [0.5, 4, 0.1], 1, v => { opt.dz = v; }),
  knob('cars', 'max cars', 'range', [0, 20000, 500], 20000, v => { opt.cars = v; }),
  knob('walk', 'pedestrians', 'range', [0, 1, 0.1], 1, v => { near.walkers.density = v; near.key = ''; }),
  knob('trees', 'trees', 'range', [0, 1, 0.1], 1, v => { looks.trees.value = v; near.key = ''; }),
  knob('hr', 'heat radius', 'range', [0.25, 3, 0.05], 1, v => { flowOpt.radius = v; }),
  knob('hi', 'heat intensity', 'range', [0.25, 4, 0.05], 1, v => { flowOpt.heat = v; }),
  knob('fp', 'flow particles', 'range', [0.1, 1, 0.05], 1, v => { flowOpt.particles = v; }),
  knob('ft', 'flow trail', 'range', [0.25, 4, 0.05], 1, v => { flowOpt.trail = v; }),
  knob('fs', 'flow sources', 'toggle', 0, false, v => { flowOpt.sources = v; }),
  knob('fc', 'flow color', 'choice', ['relative', 'absolute'], 'relative', v => { flowOpt.abs = v === 'absolute'; }),
  knob('hb', 'heat background', 'range', [0.05, 1, 0.05], 0.2, v => { opt.heat = v; apply(); }),
  knob('fb', 'flow background', 'range', [0.02, 1, 0.01], 0.2, v => { opt.flow = v; apply(); }),
  knob('tl', 'timelapse', 'range', [0.3, 5, 0.1], 1.4, v => { opt.tl = v; }),
  knob('fps', 'fps', 'toggle', 0, false, v => { fps.hidden = !v; })
]);
frame();
