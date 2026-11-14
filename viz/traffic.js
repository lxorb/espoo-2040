import * as THREE from 'three';
import { looks, COVER } from './city.js';
import { DETAIL, L } from './ground.js';

export const lengths = g => { let s = 0; return g.map((p, k) => s += k ? Math.hypot(p[0] - g[k - 1][0], p[1] - g[k - 1][1]) : 0); };
export const along = (g, cum, s, out) => {
  let k = 1;
  while (k < g.length - 1 && cum[k] < s) k++;
  const [ax, ay] = g[k - 1], [bx, by] = g[k], d = cum[k] - cum[k - 1] || 1, f = Math.min(Math.max((s - cum[k - 1]) / d, 0), 1);
  out[0] = ax + (bx - ax) * f; out[1] = ay + (by - ay) * f; out[2] = (bx - ax) / d; out[3] = (by - ay) / d;
  return out;
};
const pick = cdf => {
  const r = Math.random() * cdf[cdf.length - 1];
  let lo = 0, hi = cdf.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (cdf[m] < r) lo = m + 1; else hi = m; }
  return lo;
};
const tone = (c, r, a) => { const s = Math.min(Math.max((r - 0.3) / 0.6, 0), 1); c.setHSL(0.47 * s * Math.sqrt(s), 1, 0.5).multiplyScalar(a * (1.7 - 0.7 * s)); return 1.5 + 2 * (1 - s); };
const STEP = 20, SHUT = 0.35, GRID = 40, FINE = 10;
const W = l => Math.max(l.lanes || 1, 1) * 3.3 + 1, len = g => lengths(g).at(-1);
const full = (frag, extra = {}) => new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
  depthTest: false, depthWrite: false, transparent: true, ...extra,
  vertexShader: 'varying vec2 vU; void main() { vU = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }', fragmentShader: frag
}));
const target = (f = THREE.LinearFilter) => new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false, minFilter: f, magFilter: f });

function band(lines, y) {
  const pos = [], aS = [], aN = [], aK = [], aU = [], idx = [];
  for (const { geom: g, w, k: key, shift = 0 } of lines) {
    const cum = lengths(g), L = cum[cum.length - 1];
    g.forEach(([x, z], k) => {
      const a = g[Math.max(k - 1, 0)], b = g[Math.min(k + 1, g.length - 1)], d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      for (const s of [-1, 1]) { pos.push(x, y, -z); aS.push(s, w, shift); aN.push((b[1] - a[1]) / d, (b[0] - a[0]) / d); aK.push(...key); aU.push(cum[k], L); }
      const v = pos.length / 3 - 2;
      if (k) idx.push(v - 2, v, v - 1, v - 1, v, v + 1);
    });
  }
  const geo = new THREE.BufferGeometry();
  for (const [n, a, m] of [['position', pos, 3], ['aS', aS, 3], ['aN', aN, 2], ['aK', aK, 3], ['aU', aU, 2]]) geo.setAttribute(n, new THREE.Float32BufferAttribute(a, m));
  geo.setIndex(idx);
  return geo;
}

const BAND_VERT = `attribute vec3 aS; attribute vec2 aN; attribute vec3 aK; attribute vec2 aU; uniform float uPx; varying vec3 vK; varying vec2 vW, vU; varying float vS;
  void main() {
    vK = aK; vU = aU; vS = aS.x;
    float w = max(aS.y, 4.0 * uPx);
    vec3 p = position + vec3(aN.x, 0.0, aN.y) * (aS.x + aS.z) * 0.5 * w;
    vW = p.xz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const CLIP = 'uniform vec3 uClip; void clip() { if (dot(vec3(gl_FragCoord.xy, 1.0), uClip) < 0.0) discard; }';
const BAND_FRAG = (body, mark) => `uniform float uYear, uDim, uSat, uOy, uT, uPx; uniform vec3 uMark; varying vec3 vK; varying vec2 vW, vU; varying float vS;
  ${CLIP}
  ${COVER}
  ${DETAIL}
  float hs(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f *= f * (3.0 - 2.0 * f); return mix(mix(hs(i), hs(i + vec2(1.0, 0.0)), f.x), mix(hs(i + vec2(0.0, 1.0)), hs(i + 1.0), f.x), f.y); }
  void main() {
    clip();
    float z = exp2(ceil(log2(max(1.0, 1.25 * uPx)))), stripe = step(0.5, fract((vW.x + vW.y) / (8.0 * z))), dash = step(0.5, fract(vU.x / (16.0 * z))), e = abs(vS);
    vec3 c = vec3(0.0); float al = 1.0;
    ${body}
    c = ${mark ? 'c' : 'mix(vec3(dot(c, vec3(0.3, 0.59, 0.11))), c, uSat) * uDim'};
    gl_FragColor = vec4(c, al);
  }`;
const SURFACE = BAND_FRAG(`float wear = 1.0 - vK.x, v = vn(vW * 0.35);
    c = mix(vec3(0.15, 0.155, 0.16), vec3(0.47, 0.47, 0.45), smoothstep(0.0, 1.0, wear));
    float crack = (1.0 - smoothstep(0.0, 1.0, abs(v - 0.5) / max(fwidth(v), 1e-4))) * smoothstep(0.7, 0.9, wear) * step(0.62, vn(vW * 0.04 + 7.0));
    c *= 1.0 - 0.35 * crack * (1.0 - smoothstep(0.2, 0.5, length(fwidth(vW))));
    if (uDetail > 0.0) c *= mix(vec3(1.0), detail(vW, vW, ${L.asphalt_worn}.0), uDetail);
    al = mix(0.85, 0.7 * smoothstep(0.35, 1.0, wear), max(cover(vW), uDetail));
    if (vK.y < ${SHUT}) { float bar = step(fract(vU.x / 30.0), 0.07); c = mix(vec3(0.09), uMark, max(step(0.5, fract((vW.x + vW.y) / 5.0)), bar)); al = 0.95; }
    else if (vK.y < 0.999) { c = mix(c, uMark, stripe); al = max(al, 0.85 * stripe); }`);
const PROJECT = BAND_FRAG(`float st = vK.z, age = uYear - vK.y, f = vU.x / max(vU.y, 1.0), p = abs(vK.y - uOy) < 0.01 ? uT / 1.2 : 9.0, d = p - f;
    if (uYear < vK.x) discard;
    bool done = age >= 0.0 && d > 0.0;
    float glow = done ? exp(-d * 14.0) * step(p, 1.1) : 0.0;
    c = uMark;
    if (st > 3.5) { if (age < 0.0 && (st > 4.5 || dash < 0.5)) discard; }
    else if (!done) { if (st > 1.5 && st < 2.5 && e < 0.7) al = 0.15; else if (stripe < 0.5) discard; }
    else if (st < 0.5) discard;
    else if (e < 0.8 && glow < 0.05) { if (st < 1.5 || st > 2.5) discard; c = vec3(0.2, 0.21, 0.22); }
    c = mix(c, uMark, glow);`, true);
const KIND = { road: 1, lanes: 1, junction: 0, buslane: 3, rail: 0, bus: 0, toll: 0, parking: 0, pnr: 0 };
const mid = pts => [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];

export class Net {
  constructor(scene, net, renderer, camera, cat) {
    Object.assign(this, { scene, renderer, camera, base: net.links, nodes: net.nodes, cat: new Map((cat || []).map(p => [p.id, p])), routes: new Map(), out: [] });
    net.links.forEach((l, i) => (this.out[l.a] ||= []).push(i));
    this.byOsm = new Map();
    net.links.forEach((l, i) => this.byOsm.set(l.osm, [...(this.byOsm.get(l.osm) || []), i]));
    const twin = new Set(net.links.map(l => l.a + ',' + l.b));
    this.two = net.links.map(l => twin.has(l.b + ',' + l.a) ? 1 : 0);
    this.uni = { uPx: { value: 1 }, uRef: { value: 1 }, uClip: { value: new THREE.Vector3(0, 0, 1) }, uYear: { value: 0 }, uOy: { value: -99 }, uT: { value: 0 }, uSigma: { value: 30 }, uDim: looks.dim, uSat: looks.sat, uCover: looks.cover, uCbox: looks.cbox, uMark: looks.mark, uMats: looks.mats, uMat: looks.mat, uDetail: looks.detail };
    this.heatRT = target();
    this.heatOld = target();
    this.xf = -1e9;
    this.cache = new Map();
    this.flowRT = target(THREE.NearestFilter);
    this.flowRT2 = target(THREE.NearestFilter);
    this.heatScene = new THREE.Scene();
    this.flowScene = new THREE.Scene();
    this.heatView = full(`uniform sampler2D tMap, tOld; uniform float uRef, uHeat, uMix; varying vec2 vU; ${CLIP}
      void main() {
        clip();
        float t = log2(max(mix(texture2D(tOld, vU).r, texture2D(tMap, vU).r, uMix) * uHeat / uRef, 1e-6));
        vec3 c = mix(vec3(0.16, 0.7, 0.34), vec3(1.0, 0.88, 0.15), smoothstep(-3.4, -1.1, t));
        c = mix(c, vec3(0.95, 0.1, 0.06), smoothstep(-1.1, 0.0, t));
        c = mix(c, vec3(0.5, 0.0, 0.42), smoothstep(0.9, 1.6, t));
        gl_FragColor = vec4(c, smoothstep(-6.0, -4.2, t) * mix(0.55, 0.92, smoothstep(-3.4, 0.0, t)));
      }`);
    this.heatView.material.uniforms = { tMap: { value: this.heatRT.texture }, tOld: { value: this.heatOld.texture }, uRef: this.uni.uRef, uClip: this.uni.uClip, uHeat: { value: 1 }, uMix: { value: 1 } };
    this.flowView = full(`uniform sampler2D tMap; varying vec2 vU; ${CLIP}
      void main() { clip(); vec3 c = texture2D(tMap, vU).rgb; float m = max(c.r, max(c.g, c.b)); gl_FragColor = vec4(c / max(m, 1e-4) * (1.0 - exp(-1.6 * m)), 1.0); }`, { blending: THREE.AdditiveBlending });
    this.flowView.material.uniforms = { tMap: { value: this.flowRT.texture }, uClip: this.uni.uClip };
    this.fade = full('uniform float uA; void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, uA); }');
    this.fade.material.uniforms = { uA: { value: 0.04 } };
    this.post = new THREE.Scene();
    this.post.add(this.heatView, this.flowView);
    this.shift = full('uniform sampler2D tMap; uniform vec2 uOff; varying vec2 vU; void main() { vec2 u = vU - uOff; gl_FragColor = u == clamp(u, 0.0, 1.0) ? texture2D(tMap, u) : vec4(0.0); }', { blending: THREE.NoBlending, transparent: false });
    this.shift.material.uniforms = { tMap: { value: null }, uOff: { value: new THREE.Vector2() } };
    this.shiftScene = new THREE.Scene().add(this.shift);
    Object.assign(this, { vp: new THREE.Matrix4(), vp0: new THREE.Matrix4(), o: new THREE.Vector3(), o2: new THREE.Vector3(), rest: new THREE.Vector2(), size: new THREE.Vector2(), col: new THREE.Color(), gen: 0, kick: 0, on: false, share: 1, cut: null, pl: [], opt: { particles: 1, trail: 1, radius: 1, heat: 1, sources: false, abs: false } });
    this.postCam = new THREE.Camera();
    this.N = 60000;
    this.M = 30000;
    this.px = new Float32Array(this.N); this.py = new Float32Array(this.N); this.age = new Float32Array(this.N);
    const pool = n => ({ q: new Array(n), s: new Float32Array(n), d: new Float32Array(n), age: new Float32Array(n), lat: new Float32Array(n) });
    this.T = pool(this.M);
    this.S = pool(this.N);
    this.paths = []; this.tn = 0;
    const seg = n => {
      const g = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(1, 1));
      g.instanceCount = n;
      for (const a of ['aA', 'aC']) g.setAttribute(a, new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage));
      const m = new THREE.Mesh(g, new THREE.ShaderMaterial({ uniforms: { uPx: this.uni.uPx }, transparent: true, blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false,
        vertexShader: `attribute vec4 aA, aC; uniform float uPx; varying vec3 vC;
          void main() { vec2 d = aA.zw - aA.xy, u = d / max(length(d), 1e-6), p = mix(aA.xy, aA.zw, position.x + 0.5) + vec2(-u.y, u.x) * position.y * aC.w * uPx; vC = aC.rgb; gl_Position = projectionMatrix * viewMatrix * vec4(p.x, 1.0, -p.y, 1.0); }`,
        fragmentShader: 'varying vec3 vC; void main() { gl_FragColor = vec4(vC, 1.0); }' }));
      m.frustumCulled = false;
      return m;
    };
    this.lines = seg(this.N);
    this.thin = seg(this.M);
    this.fade.renderOrder = -1;
    this.fade.frustumCulled = false;
    this.flowScene.add(this.fade, this.lines, this.thin);
    this.rect = [0, 0, 0, 0];
    this.vlink = Int32Array.from(this.base.flatMap((l, i) => Array(l.geom.length * 2).fill(i)));
    this.roads = new THREE.Mesh(band(this.base.map((l, i) => ({ geom: l.geom, w: W(l), k: [1, 1, 0], shift: this.two[i] })), 0.5), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: BAND_VERT, fragmentShader: SURFACE, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    this.roads.renderOrder = 2;
    this.roads.frustumCulled = false;
    scene.add(this.roads);
  }

  queue() {
    const L = this.links, n = L.length, room = Float32Array.from(L, l => l.len * Math.max(l.lanes || 1, 1)), full = new Float32Array(n);
    if (this.into?.n !== n) { this.into = Object.assign([], { n }); L.forEach((l, i) => (this.into[l.b] ||= []).push(i)); }
    let wait = this.loss.slice();
    for (let pass = 0; pass < 40; pass++) {
      const next = new Float32Array(n);
      let any = false;
      for (let i = 0; i < n; i++) {
        if (!(wait[i] > 0)) continue;
        const put = Math.min(wait[i], room[i] - full[i]), over = wait[i] - put;
        full[i] += put;
        if (over < 1) continue;
        const pre = (this.into[L[i].a] || []).filter(j => L[j].a !== L[i].b), sum = pre.reduce((a, j) => a + this.flow[j], 0);
        if (!(sum > 0)) continue;
        for (const j of pre) next[j] += over * this.flow[j] / sum;
        any = true;
      }
      if (!any) break;
      wait = next;
    }
    for (let i = 0; i < n; i++) {
      const q = this.loss[i] = full[i] / room[i], r = this.ratio[i] = Math.min(this.ratio[i], 1 - 0.8 * q);
      this.speed[i] = Math.min(this.speed[i], L[i].len / this.ff[i] * r);
    }
  }

  calib(run) {
    const fl = run.years[0].flow.slice(0, this.base.length).filter(f => f > 0).sort((u, v) => u - v);
    this.scale = this.uni.uRef.value = 0.8;
    this.fref = fl[Math.floor(fl.length * 0.9)] || 1;
    this.dirty = true;
  }

  projects(run) {
    const out = [];
    run.years.forEach((y, k) => {
      for (const a of y.actions || []) {
        const p = this.cat.get(a.build);
        if (!p || a.ok === false) continue;
        const j = run.years.findIndex((z, i) => i >= k && z.open?.includes(p.id));
        out.push({ p, a: k, b: j < 0 ? k + p.years : j });
      }
    });
    return out;
  }

  blocked(x, y) {
    const s = this.grid?.get(Math.floor(x / GRID) * 65536 + Math.floor(y / GRID));
    return !!s && s.some(([ax, ay, bx, by, w]) => {
      const dx = bx - ax, dy = by - ay, t = Math.min(Math.max(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1), 0), 1);
      return Math.hypot(ax + dx * t - x, ay + dy * t - y) < w;
    });
  }

  use(run) {
    if (this.group) { this.scene.remove(this.group); this.heatScene.remove(this.heat); }
    if (!this.cache.has(run)) this.cache.set(run, this.build(run));
    Object.assign(this, this.cache.get(run), { run });
    this.S.q = this.S.q.map(q => q && q.i < this.base.length ? this.lp(q.i) : null);
    this.scene.add(this.group);
    this.heatScene.add(this.heat);
    if (!this.scale) this.calib(run);
    this.show(this.mode);
  }

  build(run) {
    const ext = [], seen = new Set();
    run.years.forEach(y => (y.new || []).forEach(l => {
      const id = JSON.stringify(l.geom);
      if (!seen.has(id)) { seen.add(id); ext.push(l); }
    }));
    const L = this.base.concat(ext), n = L.length, F = () => new Float32Array(n);
    const c = { links: L, id: this.cache.size, ff: Float32Array.from(L, l => l.len / (l.kmh / 3.6)), speed: F(), flow: F(), loss: F(), ratio: F(), cap: F().fill(1), cond: F(), cum: L.map(l => lengths(l.geom)), pl: [], paths: [], opens: new Set(), tags: [] };
    c.adj = [];
    L.forEach((l, i) => (c.adj[l.a] ||= []).push(i));
    c.oflow = new Float32Array(c.adj.length + L.length); c.iflow = new Float32Array(c.adj.length + L.length);
    c.box = L.map(l => { const xs = l.geom.map(p => p[0]), ys = l.geom.map(p => p[1]); return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; });
    const P = [], link = [];
    L.forEach((l, i) => {
      for (let k = 1; k < l.geom.length; k++) {
        const [ax, ay] = l.geom[k - 1], [bx, by] = l.geom[k], d = Math.hypot(bx - ax, by - ay);
        for (let s = Math.random() * STEP; s < d; s += STEP) { P.push(ax + (bx - ax) * s / d, ay + (by - ay) * s / d); link.push(i); }
      }
    });
    c.plink = Int32Array.from(link);
    const q = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(2, 2));
    q.instanceCount = link.length;
    q.setAttribute('aP', new THREE.InstancedBufferAttribute(new Float32Array(P), 2));
    q.setAttribute('aI', new THREE.InstancedBufferAttribute(new Float32Array(link.length), 1));
    c.heat = new THREE.Mesh(q, new THREE.ShaderMaterial({
      uniforms: this.uni, blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false, transparent: true,
      vertexShader: `attribute vec2 aP; attribute float aI; uniform float uSigma; varying vec2 vQ; varying float vI;
        void main() { float s = uSigma; vQ = position.xy * 3.0; vI = aI * ${STEP}.0 / (2.5066 * s); gl_Position = projectionMatrix * viewMatrix * vec4(aP.x + vQ.x * s, 0.0, -aP.y - vQ.y * s, 1.0); }`,
      fragmentShader: 'varying vec2 vQ; varying float vI; void main() { gl_FragColor = vec4(vI * exp(-0.5 * dot(vQ, vQ)), 0.0, 0.0, 1.0); }'
    }));
    c.heat.frustumCulled = false;
    c.dots = q.attributes.aI;
    const lines = [], stops = [], at = l => this.base[l].geom, half = g => along(g, lengths(g), len(g) / 2, [0, 0, 0, 0]);
    for (const { p, a: k, b } of this.projects(run)) {
      c.opens.add(b);
      const key = st => [k, b, st], ways = [];
      const link = (osm, st) => osm.forEach(o => (this.byOsm.get(o) || []).forEach(i => { ways.push(at(i)); lines.push({ geom: at(i), w: W(this.base[i]), k: key(st), shift: this.two[i] }); }));
      for (const m of p.works || []) link(m.osm, 0);
      for (const m of p.mod || []) link(m.osm, KIND[p.kind]);
      for (const m of p.new || []) { ways.push(m.geom); lines.push({ geom: m.geom, w: (m.oneway ? 1 : 2) * Math.max(m.lanes || 1, 1) * 3.3 + 2, k: key(2) }); }
      const route = (p.pt || []).map((t, i) => { const g = this.line(p, i, t.stops); lines.push({ geom: g, w: 3, k: key(4) }); stops.push(...t.stops.map(s => [...s, b])); return g; })[0];
      const kind = p.toll ? 'toll' : p.park ? 'park' : p.pnr ? 'pnr' : p.pt ? 'pt' : 'road', tag = (x, y) => c.tags.push({ x, y, kind, a: k, b, cost: p.cost || 0, what: p.kind, build: !!(p.works || p.new) });
      if (p.toll) for (const o of p.toll.osm) for (const i of this.byOsm.get(o) || []) {
        const g = at(i), [x, y2] = mid(g), [ax, ay] = g[0], [bx, by] = g[g.length - 1], d = Math.hypot(bx - ax, by - ay) || 1, h = W(this.base[i]) * (this.two[i] ? 1 : 0.5) + 2;
        lines.push({ geom: [[x - (by - ay) / d * h, y2 + (bx - ax) / d * h], [x + (by - ay) / d * h, y2 - (bx - ax) / d * h]], w: 2.5, k: key(5) });
        if (!c.tags.some(s => s.kind === 'toll' && Math.hypot(s.x - x, s.y - y2) < 600)) tag(x, y2);
      }
      else if (p.park) tag(...mid(p.park.area));
      else if (p.pnr) tag(...p.pnr.at);
      else {
        const g = p.pt && !p.new ? route : ways.reduce((m, g) => len(g) > len(m) ? g : m, ways[0] || route);
        if (g) tag(...half(g));
      }
    }
    c.tags.sort((u, v) => v.cost - u.cost);
    const works = new THREE.Mesh(band(lines, 0.55), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: BAND_VERT, fragmentShader: PROJECT, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.Float32BufferAttribute(stops.flatMap(([x, y]) => [x, 1.2, -y]), 3));
    sg.setAttribute('aK', new THREE.Float32BufferAttribute(stops.map(s => s[2]), 1));
    const dots = new THREE.Points(sg, new THREE.ShaderMaterial({ uniforms: this.uni, transparent: true, depthTest: false,
      vertexShader: 'attribute float aK; uniform float uYear; varying float vA; void main() { vA = step(aK, uYear); gl_PointSize = 6.0; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: CLIP + ' uniform vec3 uMark; varying float vA; void main() { clip(); float r = length(gl_PointCoord - 0.5); if (vA < 0.5 || r > 0.5) discard; gl_FragColor = vec4(r > 0.36 ? vec3(0.08) : uMark, 1.0); }' }));
    works.renderOrder = 3; dots.renderOrder = 4;
    c.group = new THREE.Group().add(works, dots);
    for (const m of [works, dots]) m.frustumCulled = false;
    return c;
  }

  line(p, q, stops) {
    const id = p.id + q;
    if (this.routes.has(id)) return this.routes.get(id);
    let g = new THREE.SplineCurve(stops.map(s => new THREE.Vector2(...s))).getPoints(stops.length * 12).map(v => [v.x, v.y]);
    if (/bus/.test(p.kind)) {
      const near = ([x, y]) => this.nodes.reduce((b, [nx, ny], i) => { const d = (nx - x) ** 2 + (ny - y) ** 2; return d < b[0] ? [d, i] : b; }, [Infinity, 0])[1], ns = stops.map(near);
      g = [stops[0]];
      for (let s = 1; s < ns.length; s++) { for (const i of this.walk(ns[s - 1], ns[s])) g.push(...this.base[i].geom); g.push(stops[s]); }
      g = g.filter((v, i) => !i || v[0] !== g[i - 1][0] || v[1] !== g[i - 1][1]);
    }
    this.routes.set(id, g);
    return g;
  }

  walk(a, b) {
    const D = new Float64Array(this.nodes.length).fill(Infinity), via = new Int32Array(this.nodes.length).fill(-1), h = [[0, a]];
    D[a] = 0;
    while (h.length) {
      const [d, u] = h[0], last = h.pop();
      if (h.length) {
        h[0] = last;
        for (let i = 0; ;) { let m = i; for (const c of [2 * i + 1, 2 * i + 2]) if (c < h.length && h[c][0] < h[m][0]) m = c; if (m === i) break; [h[i], h[m]] = [h[m], h[i]]; i = m; }
      }
      if (u === b) break;
      if (d > D[u]) continue;
      for (const i of this.out[u] || []) {
        const l = this.base[i], e = d + l.len;
        if (e >= D[l.b]) continue;
        D[l.b] = e; via[l.b] = i; h.push([e, l.b]);
        for (let k = h.length - 1; k;) { const j = (k - 1) >> 1; if (h[j][0] <= e) break; [h[k], h[j]] = [h[j], h[k]]; k = j; }
      }
    }
    const r = [];
    for (let u = b; via[u] >= 0; u = this.base[via[u]].a) r.unshift(via[u]);
    return r;
  }

  set(run, f) {
    if (run !== this.run) this.use(run);
    const ys = run.years, a = Math.min(Math.floor(f), ys.length - 1), b = Math.min(a + 1, ys.length - 1), t = f - a, A = ys[a], B = ys[b], shut = [];
    for (let i = 0; i < this.links.length; i++) {
      const ff = this.ff[i], tm = (A.time[i] ?? ff) * (1 - t) + (B.time[i] ?? ff) * t;
      this.cap[i] = A.cap?.[i] ?? 1;
      this.speed[i] = this.links[i].len / Math.max(tm, 0.1);
      this.ratio[i] = Math.min(ff / Math.max(tm, 0.1), 1);
      this.flow[i] = this.cap[i] < SHUT ? 0 : (A.flow[i] ?? 0) * (1 - t) + (B.flow[i] ?? 0) * t;
      this.loss[i] = this.flow[i] * Math.max(tm - ff, 0) / 3600 * 7.5;
      this.cond[i] = (A.cond?.[i] ?? 1) * (1 - t) + (B.cond?.[i] ?? 1) * t;
      if (this.cap[i] < SHUT) shut.push(i);
    }
    this.queue();
    const O = this.oflow.fill(0), I = this.iflow.fill(0);
    this.links.forEach((l, i) => { O[l.a] += this.flow[i]; I[l.b] += this.flow[i]; });
    let src = 0;
    this.srcCdf = Float64Array.from(O, (o, u) => src += Math.max(o - I[u], 0));
    const sk = shut.join();
    if (sk !== this.shutKey) {
      this.shutKey = sk;
      this.grid = new Map();
      for (const i of shut) {
        const g = this.links[i].geom, w = W(this.links[i]) / 2 + 3;
        for (let k = 1; k < g.length; k++) {
          const [ax, ay] = g[k - 1], [bx, by] = g[k];
          for (let x = Math.floor((Math.min(ax, bx) - w) / GRID); x <= Math.floor((Math.max(ax, bx) + w) / GRID); x++)
            for (let y = Math.floor((Math.min(ay, by) - w) / GRID); y <= Math.floor((Math.max(ay, by) + w) / GRID); y++) {
              const key = x * 65536 + y;
              this.grid.set(key, [...this.grid.get(key) || [], [ax, ay, bx, by, w]]);
            }
        }
      }
    }
    const yr = Math.floor(f + 1e-6);
    if (run === this.lastRun && yr > Math.floor(this.lastF + 1e-6) && f - this.lastF < 1.5 && this.opens.has(yr)) this.anim = { y: yr, t0: performance.now() };
    this.kick += run === this.lastRun ? Math.abs(f - this.lastF) : 1;
    this.lastRun = run; this.lastF = f;
    const base = (this.scale || 0) * 0.06;
    for (let p = 0; p < this.plink.length; p++) this.dots.array[p] = Math.max(this.loss[this.plink[p]], base);
    this.dots.needsUpdate = true;
    const k = this.roads.geometry.attributes.aK;
    for (let v = 0; v < this.vlink.length; v++) { k.array[3 * v] = this.cond[this.vlink[v]]; k.array[3 * v + 1] = this.cap[this.vlink[v]]; }
    k.needsUpdate = true;
    this.uni.uYear.value = f;
    if (Math.abs(f - (this.fieldF ?? -9)) > 0.25 || run !== this.fieldRun) { this.fieldF = f; this.fieldRun = run; this.dirty = true; }
  }

  field(rect, span, fresh) {
    const [x0, y0, x1, y1] = rect, c = Math.max(span / 300, 6), W = Math.ceil((x1 - x0) / c) + 1, H = Math.ceil((y1 - y0) / c) + 1;
    const U = new Float32Array(W * H), V = new Float32Array(W * H), D = new Float32Array(W * H), B = new Float32Array(W * H), K = new Int32Array(W * H), sg = 2 * (1.1 * c) ** 2;
    this.links.forEach((l, i) => {
      const b = this.box[i], f = this.flow[i] > 0 ? this.flow[i] : this.cap[i] < SHUT ? 0 : 0.3 * (this.fref || 1);
      if (!(f > 0) || b[0] > x1 || b[2] < x0 || b[1] > y1 || b[3] < y0) return;
      const sp = this.speed[i];
      for (let k = 1; k < l.geom.length; k++) {
        const [ax, ay] = l.geom[k - 1], [bx, by] = l.geom[k], d = Math.hypot(bx - ax, by - ay) || 1, ux = (bx - ax) / d, uy = (by - ay) / d;
        for (let s = 0; s < d; s += c / 2) {
          const x = ax + ux * s + uy * c, y = ay + uy * s - ux * c, gx = (x - x0) / c, gy = (y - y0) / c;
          for (let j = Math.max(0, Math.round(gy) - 2); j <= Math.min(H - 1, Math.round(gy) + 2); j++)
            for (let m = Math.max(0, Math.round(gx) - 2); m <= Math.min(W - 1, Math.round(gx) + 2); m++) {
              const w = f * Math.exp(-(((m - gx) * c) ** 2 + ((j - gy) * c) ** 2) / sg), o = j * W + m;
              U[o] += w * ux * sp; V[o] += w * uy * sp; D[o] += w;
              if (w > B[o]) { B[o] = w; K[o] = i; }
            }
        }
      }
    });
    const ref = 5.5 * (this.fref || 1), cdf = new Float64Array(D.length);
    let acc = 0;
    for (let o = 0; o < D.length; o++) { if (D[o] > 0) { U[o] /= D[o]; V[o] /= D[o]; } cdf[o] = acc += D[o] > ref * 0.04 ? (D[o] / ref) ** 1.5 : 0; }
    this.F = { x0, y0, c, W, H, U, V, D, K, cdf, ref, min: ref * 0.02, r: rect, span };
    if (fresh) for (let p = 0; p < this.N; p++) this.spawn(p);
  }

  trace(tiles, px) {
    const [x0, y0, x1, y1] = this.rect, P = [], v = [0, 0, 0, 0], gen = ++this.gen, out = b => b[0] > x1 || b[2] < x0 || b[1] > y1 || b[3] < y0;
    this.links.forEach((l, i) => {
      if (out(this.box[i])) return;
      if (!(this.flow[i] > 0)) { if (px < FINE && this.cap[i] >= SHUT) P.push(Object.assign(this.lp(i).z ||= { g: l.geom, cum: this.cum[i], i: -1, off: 1, w: 0.6 }, { gen })); return; }
      const m = l.geom[l.geom.length >> 1];
      v[2] = 0;
      this.sample(m[0], m[1], v);
      if (v[2] < this.F.ref * 0.04) P.push(Object.assign(this.lp(i), { gen }));
    });
    if (px < FINE) for (const t of tiles?.values() || []) {
      if (!t?.userData.res) continue;
      const [i, j] = t.name.split('_').map(Number);
      if (out([i * 1000, j * 1000, i * 1000 + 1000, j * 1000 + 1000])) continue;
      for (const f of t.userData.res) if (!f.s) P.push(Object.assign(f.p ||= { g: f.g, cum: lengths(f.g), i: -1, off: 1, w: 0.6 }, { gen }));
    }
    let s = 0;
    this.pcdf = P.map(q => s += (q.i < 0 ? q.w : 0.3 + Math.min(this.flow[q.i] / this.fref, 1.5)) * q.cum[q.cum.length - 1]);
    this.paths = P;
    this.tn = Math.min(this.M, Math.floor(s * this.opt.particles / (25 * Math.max(px, 0.5))));
  }

  lp(i) {
    return this.pl[i] ||= { g: this.links[i].geom, cum: this.cum[i], i, off: this.two[i] ?? 0, w: 1 };
  }

  rel(i) {
    return this.opt.abs ? this.speed[i] / 27.8 : this.ratio[i];
  }

  hop(u, from = -1) {
    const back = from < 0 ? -1 : this.links[from].a;
    if (!(this.oflow[u] > 0) || from >= 0 && Math.random() * this.iflow[u] > this.oflow[u]) return -1;
    const o = (this.adj[u] || []).filter(j => this.links[j].b !== back && this.flow[j] > 0);
    let r = Math.random() * o.reduce((a, j) => a + this.flow[j], 0);
    return o.find(j => (r -= this.flow[j]) < 0) ?? -1;
  }

  stream(mesh, P, n, dt, px, src) {
    const pos = mesh.geometry.attributes.aA.array, col = mesh.geometry.attributes.aC.array, e = [0, 0, 0, 0], c = this.col, k = px * 0.45 * dt, rf = Math.min(Math.max((FINE - px) / 3, 0), 1);
    let born = src && this.srcCdf.at(-1) > 0 ? Math.ceil(n * dt / 12) : 0;
    for (let p = 0; p < P.q.length; p++) {
      let q = P.q[p], s0 = P.s[p];
      P.age[p] -= dt;
      let s = q ? s0 + P.d[p] * (q.i < 0 ? 7 : this.speed[q.i]) * k : 0;
      while (src && q && s > q.cum.at(-1)) { const j = this.hop(this.links[q.i].b, q.i); s -= q.cum.at(-1); s0 = 0; q = j < 0 ? null : this.lp(j); }
      P.q[p] = q;
      if (p >= n || !q || !src && q.gen !== this.gen || P.age[p] < 0 || s < 0 || s > q.cum.at(-1)) {
        pos.fill(0, p * 4, p * 4 + 4);
        P.q[p] = null;
        const j = p >= n ? -1 : src ? born-- > 0 ? this.hop(pick(this.srcCdf)) : -1 : this.paths.length ? -2 : -1;
        if (j === -1) continue;
        P.q[p] = q = j < 0 ? this.paths[pick(this.pcdf)] : this.lp(j);
        P.s[p] = src ? 0 : Math.random() * q.cum[q.cum.length - 1];
        P.d[p] = !src && q.i < 0 && Math.random() < 0.5 ? -1 : 1;
        P.age[p] = src ? 120 : 3 + Math.random() * 5;
        P.lat[p] = Math.random() - 0.5;
        continue;
      }
      const off = src ? ((q.off ? 0.5 : 0) + P.lat[p]) * Math.max(Math.max(this.links[q.i].lanes || 1, 1) * 3.3, px * 3) : P.d[p] * (q.i < 0 ? Math.max(2, px * 1.2) : q.off * Math.max(3.3, px * 1.5));
      along(q.g, q.cum, s0, e);
      pos.set([e[0] + e[3] * off, e[1] - e[2] * off], p * 4);
      along(q.g, q.cum, s, e);
      pos.set([e[0] + e[3] * off, e[1] - e[2] * off], p * 4 + 2);
      const w = tone(c, q.i < 0 ? 1 : this.rel(q.i), (q.i < 0 ? 0.9 * rf : src ? 0.35 : 0.55) * Math.min(Math.abs(s - s0) / px, 1));
      P.s[p] = s;
      col.set([c.r, c.g, c.b, w], p * 4);
    }
    mesh.geometry.attributes.aA.needsUpdate = mesh.geometry.attributes.aC.needsUpdate = true;
  }

  spawn(p) {
    const F = this.F;
    for (let t = 0; t < 6 && (t === 0 || !this.seen(this.px[p], this.py[p])); t++) {
      const lo = pick(F.cdf);
      this.px[p] = F.x0 + (lo % F.W + Math.random() - 0.5) * F.c;
      this.py[p] = F.y0 + (Math.floor(lo / F.W) + Math.random() - 0.5) * F.c;
    }
    this.age[p] = 4 + Math.random() * 6;
  }

  seen(x, y) {
    const v = this.view;
    return x > v[0] && x < v[2] && y > v[1] && y < v[3];
  }

  keep(x, y) {
    const c = this.cut;
    return !c || ((x - c[0]) * (c[3] - c[1]) - (y - c[1]) * (c[2] - c[0])) * c[4] >= 0;
  }

  sample(x, y, out) {
    const F = this.F, gx = (x - F.x0) / F.c, gy = (y - F.y0) / F.c, i = Math.floor(gx), j = Math.floor(gy);
    if (i < 0 || j < 0 || i >= F.W - 1 || j >= F.H - 1) return false;
    const fx = gx - i, fy = gy - j, o = j * F.W + i, w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], os = [o, o + 1, o + F.W, o + F.W + 1];
    let r = 0;
    out[0] = out[1] = out[2] = 0;
    for (let k = 0; k < 4; k++) { const d = F.D[os[k]] * w[k]; out[0] += F.U[os[k]] * w[k]; out[1] += F.V[os[k]] * w[k]; out[2] += d; r += d * this.ratio[F.K[os[k]]]; }
    out[3] = r / (out[2] || 1);
    return out[2] > F.min;
  }

  draw(mode, rect, dt, px, tiles) {
    const r = this.renderer, size = r.getDrawingBufferSize(this.size);
    if (this.flowRT.width !== size.x || this.flowRT.height !== size.y) { this.flowRT.setSize(size.x, size.y); this.flowRT2.setSize(size.x, size.y); for (const t of [this.heatRT, this.heatOld]) t.setSize(Math.ceil(size.x / 2), Math.ceil(size.y / 2)); }
    this.uni.uPx.value = px;
    this.uni.uSigma.value = Math.max((5 + 19 * Math.min(Math.max((4 - px) / 3, 0), 1)) * px, 30) * this.opt.radius;
    if (!this.run || !this.on) return;
    const now = performance.now(), hv = this.heatView.material.uniforms;
    this.uni.uOy.value = this.anim ? this.anim.y : -99;
    this.uni.uT.value = this.anim ? (now - this.anim.t0) / 1000 : 0;
    if (this.run !== this.heatRun) {
      if (this.heatRun) { [this.heatRT, this.heatOld] = [this.heatOld, this.heatRT]; this.xf = now; }
      this.heatRun = this.run;
    }
    hv.tMap.value = this.heatRT.texture;
    hv.tOld.value = this.heatOld.texture;
    hv.uMix.value = Math.min((now - this.xf) / 400, 1);
    hv.uHeat.value = this.opt.heat;
    const key = this.camera.matrixWorld.elements.join() + this.camera.zoom + size.x;
    const hk = key + this.uni.uYear.value + this.id + this.opt.radius;
    if (mode === 'heat' && hk !== this.heatKey) {
      this.heatKey = hk;
      r.setRenderTarget(this.heatRT); r.setClearColor(0, 0); r.clear();
      r.render(this.heatScene, this.camera);
    }
    const src = this.opt.sources;
    this.thin.visible = !src;
    if (src && !this.srcOn) this.S.q.fill(null);
    if (mode === 'flow' && src) {
      this.pan(size);
      r.setRenderTarget(this.flowRT);
      this.stream(this.lines, this.S, Math.floor(this.N * this.opt.particles), dt, px, true);
    }
    this.srcOn = src;
    if (mode === 'flow' && !src) {
      const [x0, y0, x1, y1] = rect, w = x1 - x0, h = y1 - y0, span = Math.max(w, h), F = this.F;
      this.view = rect;
      if (this.dirty || !F || x0 < F.r[0] || y0 < F.r[1] || x1 > F.r[2] || y1 > F.r[3] || span > F.span * 1.33 || span < F.span * 0.75) {
        this.rect = [x0 - 0.3 * w, y0 - 0.3 * h, x1 + 0.3 * w, y1 + 0.3 * h];
        this.field(this.rect, span, !F);
        this.dirty = false;
      }
      const tk = `${this.rect}${this.fieldF}${[...tiles?.values() || []].filter(Boolean).length}${px < FINE}${this.opt.particles}${this.id}`;
      if (tk !== this.thinKey) { this.thinKey = tk; this.trace(tiles, px); }
      this.pan(size);
      r.setRenderTarget(this.flowRT);
      this.advect(dt, px);
      this.stream(this.thin, this.T, this.tn, dt, px, false);
    }
    if (mode === 'flow') {
      this.fade.material.uniforms.uA.value = 1 - Math.exp(-0.25 * dt / this.opt.trail - Math.min(2 * this.kick, 3));
      this.kick = 0;
      r.autoClear = false;
      r.render(this.flowScene, this.camera);
      r.autoClear = true;
    }
    r.setRenderTarget(null);
  }

  pan(size) {
    const r = this.renderer, cam = this.camera, e = cam.matrixWorld.elements;
    cam.updateMatrixWorld();
    this.vp.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    const rk = [0, 1, 2, 4, 5, 6, 8, 9, 10].map(i => Math.round(e[i] * 1e4)).join() + cam.zoom + size.x + size.y;
    if (rk !== this.rotKey) {
      this.rotKey = rk;
      this.rest.set(0, 0);
      r.setRenderTarget(this.flowRT); r.setClearColor(0, 0); r.clear();
    } else {
      const a = this.o.set(0, 0, 0).applyMatrix4(this.vp), b = this.o2.set(0, 0, 0).applyMatrix4(this.vp0);
      const dx = (a.x - b.x) / 2 * size.x + this.rest.x, dy = (a.y - b.y) / 2 * size.y + this.rest.y, ix = Math.round(dx), iy = Math.round(dy);
      this.rest.set(dx - ix, dy - iy);
      if (ix || iy) {
        this.shift.material.uniforms.tMap.value = this.flowRT.texture;
        this.shift.material.uniforms.uOff.value.set(ix / size.x, iy / size.y);
        r.setRenderTarget(this.flowRT2);
        r.render(this.shiftScene, this.postCam);
        [this.flowRT, this.flowRT2] = [this.flowRT2, this.flowRT];
        this.flowView.material.uniforms.tMap.value = this.flowRT.texture;
      }
    }
    this.vp0.copy(this.vp);
  }

  advect(dt, px) {
    const pos = this.lines.geometry.attributes.aA.array, col = this.lines.geometry.attributes.aC.array, v = [0, 0, 0, 0], c = this.col, k = px * 0.45 * dt;
    for (let p = 0; p < this.N; p++) {
      const x = this.px[p], y = this.py[p];
      this.age[p] -= dt;
      if (p >= this.N * this.opt.particles) { pos.fill(0, p * 4, p * 4 + 4); continue; }
      if (this.age[p] < 0 || !this.seen(x, y) || !this.sample(x, y, v)) { this.spawn(p); pos.fill(0, p * 4, p * 4 + 4); continue; }
      this.px[p] = x + v[0] * k; this.py[p] = y + v[1] * k;
      const w = tone(c, this.opt.abs ? Math.hypot(v[0], v[1]) / 27.8 : v[3], 0.2 * Math.min(Math.max(v[2] / this.F.ref, 0.25), 2));
      pos.set([x, y, this.px[p], this.py[p]], p * 4);
      col.set([c.r, c.g, c.b, w], p * 4);
    }
    this.lines.geometry.attributes.aA.needsUpdate = this.lines.geometry.attributes.aC.needsUpdate = true;
  }

  overlay(mode) {
    this.heatView.visible = mode === 'heat';
    this.flowView.visible = mode === 'flow';
    if (!this.run || !this.on || mode === 'cars') return;
    this.renderer.autoClear = false;
    this.renderer.render(this.post, this.postCam);
    this.renderer.autoClear = true;
  }

  show(mode) {
    this.mode = mode;
    for (const m of [this.roads, this.group]) if (m) m.visible = this.on && mode !== 'flow';
  }
}
