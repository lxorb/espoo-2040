import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const STEP = 20;
const full = (frag, extra = {}) => new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
  depthTest: false, depthWrite: false, transparent: true, ...extra,
  vertexShader: 'varying vec2 vU; void main() { vU = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }', fragmentShader: frag
}));
const target = () => new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false });

function band(links, key) {
  const pos = [], aS = [], aN = [], aK = [], idx = [];
  links.forEach((l, i) => {
    const g = l.geom, w = Math.max(l.lanes || 1, 1) * 3.3 + 2;
    g.forEach(([x, y], k) => {
      const a = g[Math.max(k - 1, 0)], b = g[Math.min(k + 1, g.length - 1)], d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      for (const s of [-1, 1]) { pos.push(x, 0.9, -y); aS.push(s, w); aN.push((b[1] - a[1]) / d, (b[0] - a[0]) / d); aK.push(key(i)); }
      const v = pos.length / 3 - 2;
      if (k) idx.push(v - 2, v, v - 1, v - 1, v, v + 1);
    });
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('aS', new THREE.Float32BufferAttribute(aS, 2));
  geo.setAttribute('aN', new THREE.Float32BufferAttribute(aN, 2));
  geo.setAttribute('aK', new THREE.Float32BufferAttribute(aK, 1));
  geo.setIndex(idx);
  return geo;
}

const BAND_VERT = `attribute vec2 aS; attribute vec2 aN; attribute float aK; uniform float uPx; varying float vK; varying vec2 vW;
  void main() {
    vK = aK;
    vec3 p = position + vec3(aN.x, 0.0, aN.y) * aS.x * 0.5 * max(aS.y, 5.0 * uPx);
    vW = p.xz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;

export class Net {
  constructor(scene, net, renderer, camera) {
    Object.assign(this, { scene, renderer, camera, base: net.links });
    this.uni = { uPx: { value: 1 }, uYear: { value: 0 }, uSigma: { value: 30 } };
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
    this.flowView = full(`uniform sampler2D tMap; varying vec2 vU; void main() { gl_FragColor = vec4(texture2D(tMap, vU).rgb, 1.0); }`, { blending: THREE.AdditiveBlending });
    this.flowView.material.uniforms = { tMap: { value: this.flowRT.texture } };
    this.fade = full('void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 0.045); }');
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
    this.speed = new Float32Array(n); this.flow = new Float32Array(n); this.loss = new Float32Array(n); this.cap = new Float32Array(n).fill(1);
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
    for (const m of [this.roads, this.works]) if (m) { this.scene.remove(m); m.geometry.dispose(); }
    this.roads = new THREE.Mesh(band(ext, i => this.open[i]), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: BAND_VERT, side: THREE.DoubleSide,
      fragmentShader: `uniform float uYear; varying float vK;
        void main() { if (uYear < vK) discard; gl_FragColor = vec4(mix(vec3(0.33, 0.35, 0.37), vec3(1.0, 0.85, 0.15), 1.0 - clamp(uYear - vK, 0.0, 1.0)), 1.0); }`
    }));
    this.works = new THREE.Mesh(band(L, () => 1), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: BAND_VERT, transparent: true, side: THREE.DoubleSide,
      fragmentShader: `varying float vK; varying vec2 vW;
        void main() { if (vK > 0.999) discard; gl_FragColor = vec4(step(0.5, fract((vW.x + vW.y) / 8.0)) * vec3(1.0, 0.5, 0.05), 0.9); }`
    }));
    this.vlink = Int32Array.from(L.flatMap((l, i) => Array(l.geom.length * 2).fill(i)));
    for (const m of [this.roads, this.works]) { m.frustumCulled = false; m.renderOrder = 2; this.scene.add(m); }
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
    }
    for (let p = 0; p < this.plink.length; p++) this.dots.array[p] = this.loss[this.plink[p]];
    this.dots.needsUpdate = true;
    const k = this.works.geometry.attributes.aK;
    for (let v = 0; v < this.vlink.length; v++) k.array[v] = this.cap[this.vlink[v]];
    k.needsUpdate = true;
    this.uni.uYear.value = f;
    if (Math.abs(f - (this.fieldF ?? -9)) > 0.25 || run !== this.fieldRun) { this.fieldF = f; this.fieldRun = run; this.dirty = true; }
  }

  events(k) {
    const y = this.run?.years[k];
    if (!y) return [];
    const nb = this.base.length, prev = this.run.years[k - 1];
    const ids = this.links.map((l, i) => i).filter(i => (i >= nb && this.open[i - nb] === k) || ((y.cap?.[i] ?? 1) !== (prev?.cap?.[i] ?? 1)));
    const segs = new Set(y.actions.map(a => a.resurface ?? a.retime).filter(v => v !== undefined));
    this.links.forEach((l, i) => { if (segs.has(l.seg)) ids.push(i); });
    return (y.events || []).map(text => {
      const named = ids.filter(i => this.links[i].name && text.includes(this.links[i].name));
      const use = named.length ? named : ids;
      if (!use.length) return { text };
      const pts = use.flatMap(i => this.links[i].geom);
      return { text, x: pts.reduce((s, p) => s + p[0], 0) / pts.length, y: pts.reduce((s, p) => s + p[1], 0) / pts.length };
    });
  }

  field(rect) {
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
    const first = !this.F;
    this.F = { x0, y0, c, W, H, U, V, D, cdf, min: max * 0.03 };
    if (first) for (let p = 0; p < this.N; p++) this.spawn(p);
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
    if (this.flowRT.width !== size.x || this.flowRT.height !== size.y) { this.flowRT.setSize(size.x, size.y); this.heatRT.setSize(Math.ceil(size.x / 4), Math.ceil(size.y / 4)); }
    this.uni.uPx.value = px;
    this.uni.uSigma.value = Math.max(24 * px, 30);
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
        if (this.dirty || !this.F || !rect.every((v, k) => Math.abs(v - o[k]) < d)) { this.rect = rect; this.field(rect); }
        this.dirty = false;
      }
      r.setRenderTarget(this.flowRT);
      if (moved) { this.camKey = key; r.setClearColor(0, 0); r.clear(); }
      this.advect(dt, px);
      r.render(this.flowScene, this.camera);
    }
    r.setRenderTarget(null);
  }

  advect(dt, px) {
    const pos = this.lines.geometry.attributes.position.array, col = this.lines.geometry.attributes.color.array, v = [0, 0, 0], c = new THREE.Color(), k = px * 4.5 * dt;
    for (let p = 0; p < this.N; p++) {
      const x = this.px[p], y = this.py[p];
      this.age[p] -= dt;
      if (this.age[p] < 0 || !this.sample(x, y, v)) { this.spawn(p); pos.fill(0, p * 6, p * 6 + 6); continue; }
      const sp = Math.hypot(v[0], v[1]);
      this.px[p] = x + v[0] * k; this.py[p] = y + v[1] * k;
      c.setHSL(0.66 * (1 - Math.min(sp * 3.6 / 100, 1)), 1, 0.5).multiplyScalar(0.55);
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
    if (this.roads) this.roads.visible = mode !== 'flow';
    if (this.works) this.works.visible = mode !== 'flow';
  }
}

const MODELS = { car: ['sedan', 4.5, 1.5, 1.85], van: ['van', 5.2, 2.2, 2.0], truck: ['delivery', 8.5, 3.4, 2.5], bus: ['van', 12, 3.1, 2.55], moto: ['kart-oobi', 2.1, 1.4, 0.8] };
const PAINT = { car: ['#e8e8e6', '#1c1d20', '#a9adb3', '#b3242a', '#2c4f8a', '#5d6267', '#d8d2c2', '#3d6b4f'], van: ['#f2f2ef', '#c9ccd0', '#2c4f8a'], truck: ['#f2f2ef', '#e07a1f', '#2c4f8a'], bus: ['#2a62b8'], moto: ['#1c1d20', '#b3242a'] };
const loads = {};

async function vehicle(t) {
  const [name, L, H, W] = MODELS[t] || MODELS.car;
  const g = await (loads[name] ||= new GLTFLoader().loadAsync(`assets/cars/${name}.glb`));
  g.scene.updateMatrixWorld(true);
  const parts = [];
  let map;
  g.scene.traverse(o => {
    if (!o.isMesh) return;
    const p = (o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone()).applyMatrix4(o.matrixWorld);
    for (const k in p.attributes) if (!['position', 'normal', 'uv'].includes(k)) p.deleteAttribute(k);
    parts.push(p);
    map = o.material.map;
  });
  const geo = mergeGeometries(parts).rotateY(Math.PI / 2);
  geo.computeBoundingBox();
  const b = geo.boundingBox, sz = b.getSize(new THREE.Vector3());
  geo.translate(-(b.min.x + b.max.x) / 2, -b.min.y, -(b.min.z + b.max.z) / 2).scale(L / sz.x, H / sz.y, W / sz.z);
  const mat = new THREE.MeshStandardMaterial({ map, roughness: 0.55 });
  mat.onBeforeCompile = s => {
    s.fragmentShader = s.fragmentShader.replace('#include <color_fragment>', `
      vec3 d = diffuseColor.rgb;
      float hi = max(max(d.r, d.g), d.b), sat = hi - min(min(d.r, d.g), d.b);
      diffuseColor.rgb = mix(d, vColor * (0.35 + 0.65 * hi), smoothstep(0.12, 0.3, sat));`);
  };
  return [geo, mat];
}

export class Cars {
  static async load(scene, meta, pts) {
    return new Cars(scene, meta, pts, await Promise.all(meta.types.map(vehicle)));
  }

  constructor(scene, meta, pts, kits) {
    this.meta = meta;
    this.pts = pts;
    this.end = Math.max(...meta.veh.map(v => v[1] + v[3]));
    this.t = 0;
    this.heading = new Float32Array(meta.veh.length);
    this.meshes = meta.types.map((t, k) => {
      const n = meta.veh.filter(v => v[0] === k).length;
      const m = new THREE.InstancedMesh(...kits[k], Math.max(n, 1));
      m.setColorAt(0, new THREE.Color());
      m.castShadow = true;
      m.frustumCulled = false;
      m.userData.paint = (PAINT[t] || PAINT.car).map(c => new THREE.Color(c));
      scene.add(m);
      return m;
    });
  }

  tick(dt) {
    this.t = (this.t + dt * this.meta.dt) % this.end;
    const o = new THREE.Object3D(), P = this.pts, count = this.meshes.map(() => 0);
    this.meta.veh.forEach(([type, start, off, n], id) => {
      const s = (this.t - start) / this.meta.dt, k = Math.floor(s);
      if (k < 0 || k >= n - 1) return;
      const a = 2 * (off + k), f = s - k;
      const x = (P[a] + (P[a + 2] - P[a]) * f) / 10, y = (P[a + 1] + (P[a + 3] - P[a + 1]) * f) / 10;
      const dx = P[a + 2] - P[a], dy = P[a + 3] - P[a + 1];
      if (dx * dx + dy * dy > 4) this.heading[id] = Math.atan2(dy, dx);
      o.position.set(x, 0.45, -y);
      o.rotation.y = this.heading[id];
      o.updateMatrix();
      const m = this.meshes[type], c = count[type]++;
      m.setMatrixAt(c, o.matrix);
      m.setColorAt(c, m.userData.paint[(id * 7919) % m.userData.paint.length]);
    });
    this.meshes.forEach((m, k) => {
      m.count = count[k];
      m.instanceMatrix.needsUpdate = true;
      if (m.instanceColor) m.instanceColor.needsUpdate = true;
    });
  }

  show(mode) {
    for (const m of this.meshes) m.visible = mode === 'cars';
  }
}
