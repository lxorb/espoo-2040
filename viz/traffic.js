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
const STEP = 12, SHUT = 0.35, FINE = 10;
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
const BAND_FRAG = (body, mark) => `uniform float uYear, uDim, uSat, uPx; uniform vec3 uMark; varying vec3 vK; varying vec2 vW, vU; varying float vS;
  ${CLIP}
  ${COVER}
  ${DETAIL}
  float hs(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f *= f * (3.0 - 2.0 * f); return mix(mix(hs(i), hs(i + vec2(1.0, 0.0)), f.x), mix(hs(i + vec2(0.0, 1.0)), hs(i + 1.0), f.x), f.y); }
  void main() {
    clip();
    float z = exp2(ceil(log2(max(1.0, 1.25 * uPx)))), stripe = step(0.5, fract((vW.x + vW.y) / (8.0 * z))), e = abs(vS);
    vec3 c = vec3(0.0); float al = 1.0;
    ${body}
    c = ${mark ? 'c' : 'mix(vec3(dot(c, vec3(0.3, 0.59, 0.11))), c, uSat) * uDim'};
    gl_FragColor = vec4(c, al);
  }`;
const SURFACE = BAND_FRAG(`float wear = 1.0 - vK.x, v = vn(vW * 0.35);
    c = mix(vec3(0.15, 0.155, 0.16), vec3(0.47, 0.47, 0.45), smoothstep(0.0, 1.0, wear));
    float crack = (1.0 - smoothstep(0.0, 1.0, abs(v - 0.5) / max(fwidth(v), 1e-4))) * smoothstep(0.7, 0.9, wear) * step(0.62, vn(vW * 0.04 + 7.0));
    c *= 1.0 - 0.35 * crack * (1.0 - smoothstep(0.2, 0.5, length(fwidth(vW))));
    if (uDetail > 0.0) c *= mix(vec3(1.0), detail(vW, vW, ${L.asphalt_worn}.0, -1.0), uDetail);
    al = mix(0.85, 0.7 * smoothstep(0.35, 1.0, wear), max(cover(vW), uDetail));
    if (vK.y < ${SHUT}) { float bar = step(fract(vU.x / 30.0), 0.07); c = mix(vec3(0.09), uMark, max(step(0.5, fract((vW.x + vW.y) / 5.0)), bar)); al = 0.95; }
    else if (vK.y < 0.999) { c = mix(c, uMark, stripe); al = max(al, 0.85 * stripe); }`);
const PROJECT = BAND_FRAG(`float st = vK.z, k = clamp((uYear - vK.y) / 0.2, 0.0, 1.0);
    c = mix(uMark, vec3(0.2, 0.21, 0.22), k);
    al = clamp((uYear - vK.x) / 0.2, 0.0, 1.0) * mix(st > 1.5 && e < 0.7 ? 0.15 : stripe, step(1.5, st) * step(e, 0.8), k);
    if (al < 0.01) discard;`, true);

export class Net {
  constructor(scene, net, renderer, camera, cat) {
    Object.assign(this, { scene, renderer, camera, base: net.links, cat: new Map((cat || []).map(p => [p.id, p])) });
    this.byOsm = new Map();
    net.links.forEach((l, i) => this.byOsm.set(l.osm, [...(this.byOsm.get(l.osm) || []), i]));
    const twin = new Set(net.links.map(l => l.a + ',' + l.b));
    this.two = net.links.map(l => twin.has(l.b + ',' + l.a) ? 1 : 0);
    this.uni = { uPx: { value: 1 }, uRef: { value: 1 }, uClip: { value: new THREE.Vector3(0, 0, 1) }, uYear: { value: 0 }, uSigma: { value: 30 }, uDim: looks.dim, uSat: looks.sat, uCover: looks.cover, uCbox: looks.cbox, uMark: looks.mark, uMats: looks.mats, uMat: looks.mat, uVar: looks.var, uDetail: looks.detail };
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
    Object.assign(this, { vp: new THREE.Matrix4(), vp0: new THREE.Matrix4(), o: new THREE.Vector3(), o2: new THREE.Vector3(), rest: new THREE.Vector2(), size: new THREE.Vector2(), col: new THREE.Color(), gen: 0, kick: 0, on: false, cut: null, opt: {} });
    this.postCam = new THREE.Camera();
    this.N = 60000;
    this.M = 30000;
    this.px = new Float32Array(this.N); this.py = new Float32Array(this.N); this.age = new Float32Array(this.N);
    this.T = { q: new Array(this.M), s: new Float32Array(this.M), d: new Float32Array(this.M), age: new Float32Array(this.M) };
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
    if (this.pre?.length !== n) { const into = []; L.forEach((l, i) => (into[l.b] ||= []).push(i)); this.pre = L.map(l => (into[l.a] || []).filter(j => L[j].a !== l.b)); }
    let wait = this.loss.slice();
    for (let pass = 0; pass < 40; pass++) {
      const next = new Float32Array(n);
      let any = false;
      for (let i = 0; i < n; i++) {
        if (!(wait[i] > 0)) continue;
        const put = Math.min(wait[i], room[i] - full[i]), over = wait[i] - put;
        full[i] += put;
        if (over < 1) continue;
        const pre = this.pre[i];
        let sum = 0;
        for (const j of pre) sum += this.flow[j];
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
    this.lastF = NaN;
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

  use(run) {
    if (this.group) { this.scene.remove(this.group); this.heatScene.remove(this.heat); }
    if (!this.cache.has(run)) this.cache.set(run, this.build(run));
    Object.assign(this, this.cache.get(run), { run });
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
    const c = { links: L, id: this.cache.size, ff: Float32Array.from(L, l => l.len / (l.kmh / 3.6)), speed: F(), flow: F(), loss: F(), ratio: F(), cap: F().fill(1), cond: F(), cum: L.map(l => lengths(l.geom)), pl: [], tags: [], proj: this.projects(run), clrKey: null };
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
    const lines = [], at = l => this.base[l].geom;
    for (const { p, a: k, b } of c.proj) {
      const key = st => [k, b, st], ways = [], hl = [];
      const link = (osm, st) => osm.forEach(o => (this.byOsm.get(o) || []).forEach(i => { ways.push(at(i)); hl.push(i); lines.push({ geom: at(i), w: W(this.base[i]), k: key(st), shift: this.two[i] }); }));
      for (const m of p.works || []) link(m.osm, 0);
      for (const m of p.mod || []) link(m.osm, p.kind === 'junction' ? 0 : 1);
      for (const m of p.new || []) { const l = { geom: m.geom, w: (m.oneway ? 1 : 2) * Math.max(m.lanes || 1, 1) * 3.3 + 2, k: key(2) }; ways.push(m.geom); hl.push(l); lines.push(l); }
      const g = ways.reduce((m, g) => len(g) > len(m) ? g : m, ways[0]);
      if (g) { const [x, y] = along(g, lengths(g), len(g) / 2, [0, 0, 0, 0]); c.tags.push({ x, y, a: k, b, cost: p.cost || 0, what: p.kind, hl }); }
    }
    c.tags.sort((u, v) => v.cost - u.cost);
    const works = new THREE.Mesh(band(lines, 0.55), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: BAND_VERT, fragmentShader: PROJECT, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    works.renderOrder = 3;
    works.frustumCulled = false;
    c.group = works;
    return c;
  }

  set(run, f) {
    if (run === this.lastRun && f === this.lastF) return;
    if (run !== this.run) this.use(run);
    const ys = run.years, a = Math.min(Math.floor(f), ys.length - 1), b = Math.min(a + 1, ys.length - 1), t = f - a, A = ys[a], B = ys[b];
    for (let i = 0; i < this.links.length; i++) {
      const ff = this.ff[i], tm = (A.time[i] ?? ff) * (1 - t) + (B.time[i] ?? ff) * t, t0 = this.opt.free ? ff : Math.max(ys[0].time[i] ?? ff, ff);
      this.cap[i] = A.cap?.[i] ?? 1;
      this.speed[i] = this.links[i].len / Math.max(tm, 0.1);
      this.ratio[i] = Math.min(t0 / Math.max(tm, 0.1), 1);
      this.flow[i] = this.cap[i] < SHUT ? 0 : (A.flow[i] ?? 0) * (1 - t) + (B.flow[i] ?? 0) * t;
      this.loss[i] = this.flow[i] * Math.max(tm - t0, 0) / 3600 * 7.5;
      this.cond[i] = (A.cond?.[i] ?? 1) * (1 - t) + (B.cond?.[i] ?? 1) * t;
    }
    this.queue();
    this.clearing(f);
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

  clearing(f) {
    const ws = this.proj.filter(q => q.a < f).flatMap(q => (q.p.new || []).map(n => [n.geom, (n.oneway ? 1 : 2) * Math.max(n.lanes || 1, 1) * 3.3 + 12])), key = ws.length;
    if (key === this.clrKey) return;
    this.clrKey = key;
    this.clr?.tex.dispose();
    this.clr = null;
    if (!ws.length) return;
    const P = ws.flatMap(([g]) => g), x0 = Math.min(...P.map(p => p[0])) - 50, y0 = Math.min(...P.map(p => p[1])) - 50;
    const R = Math.max(Math.max(...P.map(p => p[0])) + 50 - x0, Math.max(...P.map(p => p[1])) + 50 - y0) / 2048, W = Math.ceil((Math.max(...P.map(p => p[0])) + 50 - x0) / R), H = Math.ceil((Math.max(...P.map(p => p[1])) + 50 - y0) / R);
    const cv = Object.assign(document.createElement('canvas'), { width: W, height: H }), x = cv.getContext('2d');
    Object.assign(x, { strokeStyle: '#fff', lineCap: 'round', lineJoin: 'round' });
    for (const [g, w] of ws) {
      x.lineWidth = w / R;
      x.beginPath();
      g.forEach(([px, py], k) => x[k ? 'lineTo' : 'moveTo']((px - x0) / R, H - (py - y0) / R));
      x.stroke();
    }
    this.clr = { tex: new THREE.CanvasTexture(cv), box: new THREE.Vector4(x0, y0, 1 / (W * R), 1 / (H * R)) };
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

  stream(mesh, P, n, dt, px) {
    const pos = mesh.geometry.attributes.aA.array, col = mesh.geometry.attributes.aC.array, e = [0, 0, 0, 0], c = this.col, k = px * 0.45 * dt, rf = Math.min(Math.max((FINE - px) / 3, 0), 1);
    for (let p = 0; p < P.q.length; p++) {
      const q = P.q[p], s0 = P.s[p], s = q ? s0 + P.d[p] * (q.i < 0 ? 7 : this.speed[q.i]) * k : 0;
      P.age[p] -= dt;
      if (p >= n || !q || q.gen !== this.gen || P.age[p] < 0 || s < 0 || s > q.cum.at(-1)) {
        pos.fill(0, p * 4, p * 4 + 4);
        P.q[p] = p < n && this.paths.length ? this.paths[pick(this.pcdf)] : null;
        if (!P.q[p]) continue;
        P.s[p] = Math.random() * P.q[p].cum.at(-1);
        P.d[p] = P.q[p].i < 0 && Math.random() < 0.5 ? -1 : 1;
        P.age[p] = 3 + Math.random() * 5;
        continue;
      }
      const off = P.d[p] * (q.i < 0 ? Math.max(2, px * 1.2) : q.off * Math.max(3.3, px * 1.5));
      along(q.g, q.cum, s0, e);
      pos.set([e[0] + e[3] * off, e[1] - e[2] * off], p * 4);
      along(q.g, q.cum, s, e);
      pos.set([e[0] + e[3] * off, e[1] - e[2] * off], p * 4 + 2);
      const w = tone(c, q.i < 0 ? 1 : this.rel(q.i), (q.i < 0 ? 0.9 * rf : 0.55) * Math.min(Math.abs(s - s0) / px, 1));
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
    this.uni.uSigma.value = Math.max(Math.min(Math.max(14 - 1.5 * px, 5), 12) * px, 7) * this.opt.radius;
    if (!this.run || !this.on) return;
    const now = performance.now(), hv = this.heatView.material.uniforms;
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
    if (mode === 'flow') {
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
      this.stream(this.thin, this.T, this.tn, dt, px);
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
    this.renderer.autoClear = false;
    if (this.run && this.on && mode !== 'cars') this.renderer.render(this.post, this.postCam);
    if (this.group && this.on && mode === 'heat') this.renderer.render(this.group, this.camera);
    if (this.hl && this.on) this.renderer.render(this.hl, this.camera);
    this.renderer.autoClear = true;
  }

  outline(t) {
    if (t === this.hlT) return;
    this.hlT = t;
    this.hl?.children[0].geometry.dispose();
    this.hl = null;
    if (!t) return;
    const seen = new Set(), lines = t.hl.map(h => typeof h === 'number' ? { geom: this.base[h].geom, w: W(this.base[h]) * (this.two[h] ? 2 : 1) } : h).filter(({ geom: g }) => {
      const e = [g[0], g[g.length - 1]].map(p => p.join()).sort().join();
      return !seen.has(e) && seen.add(e);
    }).map(({ geom, w }) => ({ geom, w: w + 5, k: [0, 0, 0] }));
    const m = new THREE.Mesh(band(lines, 0.6), new THREE.ShaderMaterial({ uniforms: this.uni, vertexShader: BAND_VERT.replace('max(aS.y, 4.0 * uPx)', 'aS.y + 10.0 * uPx'), transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
      fragmentShader: `uniform float uPx; uniform vec3 uMark; varying vec2 vU; varying float vS; ${CLIP}
        void main() { clip(); if ((1.0 - abs(vS)) / max(fwidth(vS), 1e-5) > 2.0 || fract(vU.x / (12.0 * uPx)) > 0.6) discard; gl_FragColor = vec4(uMark, 1.0); }` }));
    m.frustumCulled = false;
    this.hl = new THREE.Scene().add(m);
  }

  show(mode) {
    this.mode = mode;
    for (const m of [this.roads, this.group]) if (m) m.visible = this.on && mode !== 'flow';
  }
}
