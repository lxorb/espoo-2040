import * as THREE from 'three';
import { looks } from './city.js';

export const lengths = g => { let s = 0; return g.map((p, k) => s += k ? Math.hypot(p[0] - g[k - 1][0], p[1] - g[k - 1][1]) : 0); };
const STEP = 20;
const full = (frag, extra = {}) => new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
  depthTest: false, depthWrite: false, transparent: true, ...extra,
  vertexShader: 'varying vec2 vU; void main() { vU = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }', fragmentShader: frag
}));
const target = () => new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false });

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
const BAND_FRAG = (body) => `uniform float uYear, uDim, uSat; varying vec3 vK; varying vec2 vW, vU; varying float vS;
  void main() {
    float stripe = step(0.5, fract((vW.x + vW.y) / 8.0)), dash = step(0.5, fract(vU.x / 16.0)), e = abs(vS);
    vec3 c = vec3(0.0); float al = 1.0;
    ${body}
    c = mix(vec3(dot(c, vec3(0.3, 0.59, 0.11))), c, uSat) * uDim;
    gl_FragColor = vec4(c, al);
  }`;
const SURFACE = BAND_FRAG(`float h = fract(sin(dot(floor(vW / 2.5), vec2(12.9898, 78.233))) * 43758.5453), wear = 1.0 - vK.x;
    c = mix(vec3(0.17, 0.18, 0.19), vec3(0.56, 0.56, 0.53), wear) * (1.0 - 0.22 * wear * h);
    al = 0.8;
    if (vK.y < 0.999) { c = stripe * vec3(1.0, 0.5, 0.05); al = 0.95; }`);
const PROJECT = BAND_FRAG(`float st = vK.z, age = uYear - vK.y, f = vU.x / max(vU.y, 1.0), d = age / 0.7 - f;
    if (uYear < vK.x) discard;
    bool done = age >= 0.0 && d > 0.0;
    float glow = done ? exp(-d * 8.0) * step(age, 1.2) : 0.0;
    if (st > 4.5) { if (age < 0.0) discard; c = vec3(0.9, 0.15, 0.12); }
    else if (st > 3.5) { if (age < 0.0 && dash < 0.5) discard; c = vec3(0.25, 0.55, 1.0); }
    else if (!done) {
      if (st > 1.5 && st < 2.5) { if (e < 0.7) { c = vec3(1.0, 0.5, 0.05); al = 0.3; } else { if (dash < 0.5) discard; c = vec3(1.0, 0.5, 0.05); } }
      else c = stripe * vec3(1.0, 0.5, 0.05);
    } else if (st < 0.5) discard;
    else if (st > 2.5) { if (e > 0.6 && glow < 0.05) discard; c = vec3(0.8, 0.12, 0.1); al = 0.85; }
    else { if (e < 0.75 && st < 1.5 && glow < 0.05) discard; c = e < 0.75 ? vec3(0.2, 0.21, 0.22) : vec3(1.0, 0.82, 0.1); }
    c = mix(c, vec3(1.0, 0.95, 0.6), glow);`);
const KIND = { road: 1, lanes: 1, junction: 1, buslane: 3, rail: 0, bus: 0, toll: 0, parking: 0, pnr: 0 };
const CAT = { road: 'works', lanes: 'works', junction: 'works', rail: 'pt', bus: 'pt', buslane: 'pt', toll: 'toll', parking: 'toll', pnr: 'toll' };
const mid = pts => [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];

export class Net {
  constructor(scene, net, renderer, camera, cat) {
    Object.assign(this, { scene, renderer, camera, base: net.links, segs: net.segs || [], cat: new Map((cat || []).map(p => [p.id, p])) });
    this.byOsm = new Map();
    net.links.forEach((l, i) => this.byOsm.set(l.osm, [...(this.byOsm.get(l.osm) || []), i]));
    const twin = new Set(net.links.map(l => l.a + ',' + l.b));
    this.two = net.links.map(l => twin.has(l.b + ',' + l.a) ? 1 : 0);
    this.uni = { uPx: { value: 1 }, uYear: { value: 0 }, uSigma: { value: 30 }, uDim: looks.dim, uSat: looks.sat };
    this.heatRT = target();
    this.flowRT = target();
    this.heatScene = new THREE.Scene();
    this.flowScene = new THREE.Scene();
    this.heatView = full(`uniform sampler2D tMap; uniform float uScale; varying vec2 vU;
      void main() {
        float t = texture2D(tMap, vU).r / uScale;
        vec3 c = t < 0.5 ? mix(vec3(0.2, 0.85, 0.3), vec3(1.0, 0.9, 0.15), t / 0.5) : mix(vec3(1.0, 0.9, 0.15), vec3(0.95, 0.12, 0.1), clamp((t - 0.5) / 0.5, 0.0, 1.0));
        gl_FragColor = vec4(c, smoothstep(0.04, 0.3, t) * 0.78);
      }`);
    this.heatView.material.uniforms = { tMap: { value: this.heatRT.texture }, uScale: { value: 1 } };
    this.flowView = full(`uniform sampler2D tMap; varying vec2 vU;
      void main() { vec3 c = texture2D(tMap, vU).rgb; float m = max(c.r, max(c.g, c.b)); gl_FragColor = vec4(c / max(m, 1e-4) * (1.0 - exp(-1.6 * m)), 1.0); }`, { blending: THREE.AdditiveBlending });
    this.flowView.material.uniforms = { tMap: { value: this.flowRT.texture } };
    this.fade = full('uniform float uA; void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, uA); }');
    this.fade.material.uniforms = { uA: { value: 0.04 } };
    this.post = new THREE.Scene();
    this.post.add(this.heatView, this.flowView);
    this.postCam = new THREE.Camera();
    this.N = 60000;
    this.px = new Float32Array(this.N); this.py = new Float32Array(this.N); this.age = new Float32Array(this.N);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.N * 6), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.N * 6), 3).setUsage(THREE.DynamicDrawUsage));
    this.lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthTest: false }));
    this.lines.frustumCulled = false;
    this.fade.renderOrder = -1;
    this.fade.frustumCulled = false;
    this.flowScene.add(this.fade, this.lines);
    this.rect = [0, 0, 0, 0];
    this.camKey = '';
  }

  use(run) {
    this.run = run;
    const ext = [], seen = new Set();
    this.open = [];
    run.years.forEach((y, k) => (y.new || []).forEach(l => {
      const id = JSON.stringify(l.geom);
      if (!seen.has(id)) { seen.add(id); ext.push(l); this.open.push(k); }
    }));
    this.links = this.base.concat(ext);
    const L = this.links, n = L.length, nb = this.base.length;
    this.ff = Float32Array.from(L, l => l.len / (l.kmh / 3.6));
    this.speed = new Float32Array(n); this.flow = new Float32Array(n); this.loss = new Float32Array(n); this.cap = new Float32Array(n).fill(1); this.cond = new Float32Array(n);
    this.box = L.map(l => { const xs = l.geom.map(p => p[0]), ys = l.geom.map(p => p[1]); return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; });
    const P = [], link = [];
    L.forEach((l, i) => {
      for (let k = 1; k < l.geom.length; k++) {
        const [ax, ay] = l.geom[k - 1], [bx, by] = l.geom[k], d = Math.hypot(bx - ax, by - ay);
        for (let s = Math.random() * STEP; s < d; s += STEP) { P.push(ax + (bx - ax) * s / d, ay + (by - ay) * s / d); link.push(i); }
      }
    });
    this.plink = Int32Array.from(link);
    const q = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(2, 2));
    q.instanceCount = link.length;
    q.setAttribute('aP', new THREE.InstancedBufferAttribute(new Float32Array(P), 2));
    q.setAttribute('aI', new THREE.InstancedBufferAttribute(new Float32Array(link.length), 1));
    this.heatScene.clear();
    const dots = new THREE.Mesh(q, new THREE.ShaderMaterial({
      uniforms: this.uni, blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false, transparent: true,
      vertexShader: `attribute vec2 aP; attribute float aI; uniform float uSigma; varying vec2 vQ; varying float vI;
        void main() { vQ = position.xy * 3.0; vI = aI * ${STEP}.0 / (2.5066 * uSigma); gl_Position = projectionMatrix * viewMatrix * vec4(aP.x + vQ.x * uSigma, 0.0, -aP.y - vQ.y * uSigma, 1.0); }`,
      fragmentShader: 'varying vec2 vQ; varying float vI; void main() { gl_FragColor = vec4(vI * exp(-0.5 * dot(vQ, vQ)), 0.0, 0.0, 1.0); }'
    }));
    dots.frustumCulled = false;
    this.heatScene.add(dots);
    this.dots = q.attributes.aI;
    for (const m of [this.roads, this.works, this.stops]) if (m) { this.scene.remove(m); m.geometry.dispose(); }
    const W = l => Math.max(l.lanes || 1, 1) * 3.3 + 1, lines = [], stops = [], at = l => this.base[l].geom;
    this.signs = []; this.ticks = []; this.cards = [];
    run.years.forEach((y, k) => {
      const segs = new Set(y.actions.map(a => a.resurface ?? a.retime).filter(v => v !== undefined));
      if (segs.size) this.ticks.push({ k, cat: 'care', at: mid(this.segs.filter(s => segs.has(s.id)).flatMap(s => s.links.map(i => mid(at(i))))) });
      for (const a of y.actions) {
        const p = this.cat.get(a.build);
        if (!p) continue;
        const b = k + p.years, pts = [], key = st => [k, b, st];
        const link = (osm, st) => osm.forEach(o => (this.byOsm.get(o) || []).forEach(i => { lines.push({ geom: at(i), w: W(this.base[i]), k: key(st), shift: this.two[i] }); pts.push(mid(at(i))); }));
        for (const m of p.works || []) link(m.osm, 0);
        for (const m of p.mod || []) link(m.osm, KIND[p.kind]);
        for (const n of p.new || []) { lines.push({ geom: n.geom, w: (n.oneway ? 1 : 2) * Math.max(n.lanes || 1, 1) * 3.3 + 2, k: key(2) }); pts.push(...n.geom); }
        for (const t of p.pt || []) { lines.push({ geom: t.stops, w: 3, k: key(4) }); stops.push(...t.stops.map(s => [...s, b])); pts.push(...t.stops); }
        if (p.toll) for (const o of p.toll.osm) for (const i of this.byOsm.get(o) || []) {
          const g = at(i), [x, y2] = mid(g), [ax, ay] = g[0], [bx, by] = g[g.length - 1], d = Math.hypot(bx - ax, by - ay) || 1, h = W(this.base[i]) * (this.two[i] ? 1 : 0.5) + 2;
          lines.push({ geom: [[x - (by - ay) / d * h, y2 + (bx - ax) / d * h], [x + (by - ay) / d * h, y2 - (bx - ax) / d * h]], w: 2.5, k: key(5) });
          pts.push([x, y2]);
          if (!this.signs.some(s => s.a === b && Math.hypot(s.x - x, s.y - y2) < 600)) this.signs.push({ text: '€', x, y: y2, a: b });
        }
        if (p.park) { const [x, y2] = mid(p.park.area); this.signs.push({ text: 'P', x, y: y2, a: b }); pts.push([x, y2]); }
        if (p.pnr) { this.signs.push({ text: 'P', x: p.pnr.at[0], y: p.pnr.at[1], a: b }); pts.push(p.pnr.at); }
        const c = mid(pts.length ? pts : [[0, 0]]);
        this.ticks.push({ k, cat: CAT[p.kind] || 'works', at: c });
        if (!this.cards[k] || this.cards[k].cost < p.cost) this.cards[k] = { cost: p.cost, x: c[0], y: c[1], text: `${p.name}\n${p.cost} M EUR\n${p.years} ${p.years > 1 ? 'years' : 'year'}` };
      }
    });
    this.roads = new THREE.Mesh(band(this.base.map((l, i) => ({ geom: l.geom, w: W(l), k: [1, 1, 0], shift: this.two[i] })), 0.5), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: BAND_VERT, fragmentShader: SURFACE, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    this.works = new THREE.Mesh(band(lines, 0.55), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: BAND_VERT, fragmentShader: PROJECT, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.Float32BufferAttribute(stops.flatMap(([x, y]) => [x, 1.2, -y]), 3));
    sg.setAttribute('aK', new THREE.Float32BufferAttribute(stops.map(s => s[2]), 1));
    this.stops = new THREE.Points(sg, new THREE.ShaderMaterial({ uniforms: this.uni, transparent: true, depthTest: false,
      vertexShader: 'attribute float aK; uniform float uYear; varying float vA; void main() { vA = step(aK, uYear); gl_PointSize = 11.0; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'varying float vA; void main() { float r = length(gl_PointCoord - 0.5); if (vA < 0.5 || r > 0.5) discard; gl_FragColor = vec4(r > 0.32 ? vec3(0.25, 0.55, 1.0) : vec3(1.0), 1.0); }' }));
    this.vlink = Int32Array.from(this.base.flatMap((l, i) => Array(l.geom.length * 2).fill(i)));
    this.roads.renderOrder = 2; this.works.renderOrder = 3; this.stops.renderOrder = 4;
    for (const m of [this.roads, this.works, this.stops]) { m.frustumCulled = false; this.scene.add(m); }
    this.show(this.mode);
    if (!this.scale) {
      const all = [];
      run.years.forEach((y, k) => { this.set(run, k); for (const v of this.loss) if (v > 0) all.push(v); });
      all.sort((a, b) => a - b);
      this.scale = 2 * (all[Math.floor(all.length * 0.98)] || 1);
      this.heatView.material.uniforms.uScale.value = this.scale;
    }
  }

  set(run, f) {
    if (run !== this.run) this.use(run);
    const ys = run.years, a = Math.min(Math.floor(f), ys.length - 1), b = Math.min(a + 1, ys.length - 1), t = f - a, A = ys[a], B = ys[b];
    for (let i = 0; i < this.links.length; i++) {
      const ff = this.ff[i], tm = (A.time[i] ?? ff) * (1 - t) + (B.time[i] ?? ff) * t;
      this.speed[i] = this.links[i].len / Math.max(tm, 0.1);
      this.flow[i] = (A.flow[i] ?? 0) * (1 - t) + (B.flow[i] ?? 0) * t;
      this.loss[i] = this.flow[i] * Math.max(tm - ff, 0) / 3.6 / this.links[i].len;
      this.cap[i] = A.cap?.[i] ?? 1;
      this.cond[i] = (A.cond?.[i] ?? 1) * (1 - t) + (B.cond?.[i] ?? 1) * t;
    }
    for (let p = 0; p < this.plink.length; p++) this.dots.array[p] = this.loss[this.plink[p]];
    this.dots.needsUpdate = true;
    const k = this.roads.geometry.attributes.aK;
    for (let v = 0; v < this.vlink.length; v++) { k.array[3 * v] = this.cond[this.vlink[v]]; k.array[3 * v + 1] = this.cap[this.vlink[v]]; }
    k.needsUpdate = true;
    this.uni.uYear.value = f;
    if (Math.abs(f - (this.fieldF ?? -9)) > 0.25 || run !== this.fieldRun) { this.fieldF = f; this.fieldRun = run; this.dirty = true; }
  }

  field(rect, fresh) {
    const [x0, y0, x1, y1] = rect, c = Math.max(Math.max(x1 - x0, y1 - y0) / 300, 6), W = Math.ceil((x1 - x0) / c) + 1, H = Math.ceil((y1 - y0) / c) + 1;
    const U = new Float32Array(W * H), V = new Float32Array(W * H), D = new Float32Array(W * H), sg = 2 * (1.1 * c) ** 2;
    this.links.forEach((l, i) => {
      const b = this.box[i], f = this.flow[i];
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
            }
        }
      }
    });
    let max = 0;
    for (let o = 0; o < D.length; o++) { if (D[o] > 0) { U[o] /= D[o]; V[o] /= D[o]; } max = Math.max(max, D[o]); }
    const cdf = new Float32Array(D.length);
    let acc = 0;
    for (let o = 0; o < D.length; o++) cdf[o] = acc += D[o] > max * 0.05 ? D[o] : 0;
    this.F = { x0, y0, c, W, H, U, V, D, cdf, min: max * 0.03 };
    if (fresh) for (let p = 0; p < this.N; p++) this.spawn(p);
  }

  spawn(p) {
    const F = this.F, r = Math.random() * F.cdf[F.cdf.length - 1];
    let lo = 0, hi = F.cdf.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (F.cdf[m] < r) lo = m + 1; else hi = m; }
    this.px[p] = F.x0 + (lo % F.W + Math.random() - 0.5) * F.c;
    this.py[p] = F.y0 + (Math.floor(lo / F.W) + Math.random() - 0.5) * F.c;
    this.age[p] = 4 + Math.random() * 6;
  }

  sample(x, y, out) {
    const F = this.F, gx = (x - F.x0) / F.c, gy = (y - F.y0) / F.c, i = Math.floor(gx), j = Math.floor(gy);
    if (i < 0 || j < 0 || i >= F.W - 1 || j >= F.H - 1) return false;
    const fx = gx - i, fy = gy - j, o = j * F.W + i, w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], os = [o, o + 1, o + F.W, o + F.W + 1];
    out[0] = out[1] = out[2] = 0;
    for (let k = 0; k < 4; k++) { out[0] += F.U[os[k]] * w[k]; out[1] += F.V[os[k]] * w[k]; out[2] += F.D[os[k]] * w[k]; }
    return out[2] > F.min;
  }

  draw(mode, rect, dt, px) {
    const r = this.renderer, size = r.getDrawingBufferSize(new THREE.Vector2());
    if (this.flowRT.width !== size.x || this.flowRT.height !== size.y) { this.flowRT.setSize(size.x, size.y); this.heatRT.setSize(Math.ceil(size.x / 2), Math.ceil(size.y / 2)); }
    this.uni.uPx.value = px;
    this.uni.uSigma.value = Math.max((5 + 19 * Math.min(Math.max((4 - px) / 3, 0), 1)) * px, 30);
    if (!this.run) return;
    const key = this.camera.matrixWorld.elements.join() + this.camera.zoom + size.x;
    if (mode === 'heat' && key + this.uni.uYear.value + this.run.who !== this.heatKey) {
      this.heatKey = key + this.uni.uYear.value + this.run.who;
      r.setRenderTarget(this.heatRT); r.setClearColor(0, 0); r.clear();
      r.render(this.heatScene, this.camera);
    }
    if (mode === 'flow') {
      const moved = key !== this.camKey;
      if (moved || this.dirty) {
        const o = this.rect, d = Math.max(rect[2] - rect[0], rect[3] - rect[1]) * 0.1;
        const fresh = !this.F || !rect.every((v, k) => Math.abs(v - o[k]) < d);
        if (this.dirty || fresh) { this.rect = rect; this.field(rect, fresh); }
        this.dirty = false;
      }
      r.setRenderTarget(this.flowRT);
      if (moved) { this.camKey = key; r.setClearColor(0, 0); r.clear(); }
      this.advect(dt, px);
      this.fade.material.uniforms.uA.value = 1 - Math.exp(-0.25 * dt);
      r.autoClear = false;
      r.render(this.flowScene, this.camera);
      r.autoClear = true;
    }
    r.setRenderTarget(null);
  }

  advect(dt, px) {
    const pos = this.lines.geometry.attributes.position.array, col = this.lines.geometry.attributes.color.array, v = [0, 0, 0], c = new THREE.Color(), k = px * 0.45 * dt;
    for (let p = 0; p < this.N; p++) {
      const x = this.px[p], y = this.py[p];
      this.age[p] -= dt;
      if (this.age[p] < 0 || !this.sample(x, y, v)) { this.spawn(p); pos.fill(0, p * 6, p * 6 + 6); continue; }
      const sp = Math.hypot(v[0], v[1]);
      this.px[p] = x + v[0] * k; this.py[p] = y + v[1] * k;
      c.setHSL(0.66 * (1 - Math.min(sp * 3.6 / 100, 1)), 1, 0.5).multiplyScalar(0.3);
      pos.set([x, 1, -y, this.px[p], 1, -this.py[p]], p * 6);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], p * 6);
    }
    this.lines.geometry.attributes.position.needsUpdate = this.lines.geometry.attributes.color.needsUpdate = true;
  }

  overlay(mode) {
    this.heatView.visible = mode === 'heat';
    this.flowView.visible = mode === 'flow';
    if (!this.run || mode === 'cars') return;
    this.renderer.autoClear = false;
    this.renderer.render(this.post, this.postCam);
    this.renderer.autoClear = true;
  }

  show(mode) {
    this.mode = mode;
    for (const m of [this.roads, this.works, this.stops]) if (m) m.visible = mode !== 'flow';
  }
}
