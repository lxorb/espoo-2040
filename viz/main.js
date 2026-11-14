import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { City, HYPE, looks } from './city.js';
import { Net, Cars } from './traffic.js';

const STATIONS = [['Kivenlahti', -1201, 1072], ['Espoonlahti', -205, 361], ['Soukka', 603, -544], ['Kaitaa', 1797, 312], ['Finnoo', 2981, 656],
  ['Matinkylä', 4518, 1343], ['Niittykumpu', 5884, 2487], ['Urheilupuisto', 6902, 2920], ['Tapiola', 8165, 2929], ['Aalto-yliopisto', 9295, 3944], ['Keilaniemi', 9590, 2952]];
const RUNS = ['agent', 'reference', 'nothing'];
const SKY = { cars: '#2b2e31', heat: '#1d1f22', flow: '#000000' };
const q = new URLSearchParams(location.search);
const state = { mode: q.get('mode') || 'cars', who: q.get('run'), f: +(q.get('year') || 0), play: false, run: null, speed: 1, shown: '' };

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: q.has('shot') });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 60000);
camera.zoom = +(q.get('zoom') || 1.6);
const controls = new MapControls(camera, renderer.domElement);
controls.enableDamping = false;
controls.minZoom = 0.03;
controls.maxZoom = 30;
controls.maxPolarAngle = 1.25;
controls.zoomToCursor = true;
const [tx, ty] = (q.get('at') || HYPE.join(',')).split(',').map(Number);
controls.target.set(tx, 0, -ty);
camera.position.set(tx - 7000, 7000, -ty + 7000);
controls.update();

const hemi = new THREE.HemisphereLight('#e4ebf2', '#6d6a62', 1.2);
const sun = new THREE.DirectionalLight('#fff3e2', 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.6;
scene.add(hemi, sun, sun.target);

const city = new City(scene);
let net, cars;

fetch('data/net.json').then(r => r.json()).then(n => { net = new Net(scene, n, renderer, camera); apply(); runList.then(loadRun); });
Promise.all([fetch('data/cars.json').then(r => r.json()), fetch('data/cars.bin').then(r => r.arrayBuffer())])
  .then(([m, b]) => Cars.load(scene, m, new Int32Array(b))).then(c => { cars = c; apply(); }).catch(e => console.log(e));
const hypeMats = [], fine = [];
let lod = 0, coarse = false;
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
      for (const m of [o.material].flat()) { m.userData.base = m.color.clone(); hypeMats.push(m); }
      if (!footprint) return;
      const a = o.geometry.attributes.position, w = new THREE.Vector3(), pts = [];
      for (let i = 0; i < a.count; i++) { w.fromBufferAttribute(a, i).applyMatrix4(o.matrixWorld); pts.push([w.x, -w.z]); }
      polys.push(hull(pts));
    });
    scene.add(g.scene);
    fine.push(g.scene);
    g.scene.scale.y = Math.max(1 - lod, 0.001);
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

const runs = {};
const runList = fetch('data/runs/index.json').then(r => r.json()).catch(() => RUNS).then(list => {
  for (const r of list) document.getElementById('runs').insertAdjacentHTML('beforeend', `<button data-v="${r}">${r}</button>`);
  state.who ||= list[0];
  sync();
});
function loadRun() {
  const who = state.who;
  (runs[who] ||= fetch(`data/runs/${who}.json`).then(r => r.json())).then(run => {
    if (who !== state.who) return;
    state.run = run;
    year.max = run.years.length - 1;
    state.f = Math.min(state.f, year.max);
    net && net.set(run, state.f);
    sync();
  }).catch(() => { document.querySelector(`#runs [data-v=${who}]`).disabled = true; });
}

const year = document.getElementById('year'), ylab = document.getElementById('ylab'), play = document.getElementById('play');
function sync() {
  for (const b of document.querySelectorAll('#modes button')) b.classList.toggle('on', b.dataset.v === state.mode);
  for (const b of document.querySelectorAll('#runs button')) b.classList.toggle('on', b.dataset.v === state.who);
  play.classList.toggle('on', state.play);
  year.value = state.f;
  ylab.textContent = (state.run ? state.run.y0 : 2026) + Math.round(state.f);
}
function apply() {
  scene.background = new THREE.Color(SKY[state.mode]);
  looks.dim.value = { cars: 1, heat: 0.8, flow: 0.07 }[state.mode];
  looks.sat.value = { cars: 1, heat: 0.5, flow: 0.25 }[state.mode];
  for (const m of hypeMats) m.userData.base && m.color.copy(m.userData.base).multiplyScalar(looks.dim.value);
  hemi.intensity = state.mode === 'flow' ? 0.5 : 1.2;
  net && net.show(state.mode);
  cars && cars.show(state.mode);
  sync();
}
document.getElementById('modes').onclick = e => { if (e.target.dataset.v) { state.mode = e.target.dataset.v; apply(); } };
document.getElementById('runs').onclick = e => { if (e.target.dataset.v) { state.who = e.target.dataset.v; loadRun(); sync(); } };
year.oninput = () => { state.f = +year.value; state.run && net && net.set(state.run, state.f); sync(); };
document.getElementById('speed').onchange = e => { state.speed = +e.target.value; };
play.onclick = () => { state.play = !state.play; if (state.play && state.f >= +year.max) state.f = 0; sync(); };
addEventListener('keydown', e => {
  if (e.key === ' ') play.onclick();
  const m = { 1: 'cars', 2: 'heat', 3: 'flow' }[e.key];
  if (m) { state.mode = m; apply(); }
});

const news = document.getElementById('news');
let events = [];
function eventTags() {
  const k = Math.floor(state.f + 1e-6), key = state.who + k;
  if (!net || !state.run || key === state.shown) return;
  state.shown = key;
  for (const e of events) e.el.remove();
  news.replaceChildren();
  const at = {};
  events = net.events(k).map(({ text, x, y }) => {
    const el = document.createElement('div');
    el.className = 'lab ev';
    el.textContent = text;
    if (x === undefined) { news.append(el); return null; }
    document.getElementById('labels').append(el);
    const n = at[x + ',' + y] = (at[x + ',' + y] || 0) + 1;
    return { el, p: new THREE.Vector3(x, 20, -y), dy: (n - 1) * 30 };
  }).filter(Boolean);
}

const labels = [['HYPE', ...HYPE, 40, 'big'], ...STATIONS.map(([n, x, y]) => [n.toUpperCase(), x, y, 12, 'st'])].map(([text, x, y, h, cls]) => {
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

const clock = new THREE.Clock();
let last = '';
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (fly) flying(dt);
  coarse = camera.zoom < (coarse ? 1.0 : 0.8);
  const lt = Math.min(Math.max(lod + (coarse ? dt : -dt) / 0.4, 0), 1);
  if (lt !== lod) {
    lod = lt;
    city.setLod(lod);
    for (const g of fine) { g.scale.y = Math.max(1 - lod, 0.001); g.visible = lod < 1; }
  }
  controls.update();
  const px = 1 / camera.zoom;
  const [x0, y0, x1, y1] = view();
  city.update(x0 - 200, y0 - 200, x1 + 200, y1 + 200);
  city.show(state.mode, camera.zoom);
  const span = Math.min(Math.max(x1 - x0, y1 - y0) * 0.6, 3000), t = controls.target;
  sun.position.set(t.x - 0.45 * 4000, 4000 * 0.8, t.z + 0.75 * 4000);
  sun.target.position.copy(t);
  Object.assign(sun.shadow.camera, { left: -span, right: span, top: span, bottom: -span, near: 100, far: 12000 });
  sun.shadow.camera.updateProjectionMatrix();
  if (state.play && state.run) {
    state.f = Math.min(state.f + dt / 1.4, +year.max);
    if (state.f >= +year.max) state.play = false;
    net && net.set(state.run, state.f);
    sync();
  }
  if (cars && state.mode === 'cars') cars.tick(dt * state.speed);
  net && net.draw(state.mode, [x0, y0, x1, y1], dt * state.speed, px);
  eventTags();
  const w = innerWidth, h = innerHeight;
  for (const l of [...labels, ...events]) {
    v.copy(l.p).project(camera);
    const on = Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 && (!l.el.classList.contains('st') || camera.zoom > 0.12);
    l.el.style.display = on ? '' : 'none';
    if (on) l.el.style.transform = `translate(${(v.x + 1) / 2 * w}px, ${(1 - v.y) / 2 * h - 8 - (l.dy || 0)}px) translate(-50%, -100%)`;
  }
  renderer.render(scene, camera);
  net && net.overlay(state.mode);
  const s = `${city.loaded.size}`;
  if (s !== last) { last = s; document.body.dataset.tiles = s; }
  requestAnimationFrame(frame);
}
apply();
frame();
