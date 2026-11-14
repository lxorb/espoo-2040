import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { looks, patch, up, DITHER, HYPE } from './city.js';
import { lengths, along } from './traffic.js';

export const caps = { ground: 64, texels: 40e6, cache: 80e6, sprites: 10000, walkers: 300, props: 3000, loads: 4 };
const GROUPS = [['birch', 'linden'], ['pine', 'spruce'], ['linden', 'linden', 'birch'], ['pine', 'spruce', 'birch'], ['pine', 'spruce'], ['shrub']];
const SHADOW = [0.51, 0.86];
const v = new THREE.Vector3();
export const stamp = (c, ...x) => [c.position.x / 8, c.position.z / 8, c.zoom * 50, c.quaternion.y * 500, c.quaternion.w * 500, ...x].reduce((h, a) => Math.imul(h ^ Math.round(a), 16777619), 2166136261);

function quad() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 0, -1], 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 1, 1, 1, 1, 0, 0, 0], 2));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  return up(g);
}

class Ground {
  constructor(scene, renderer) {
    Object.assign(this, { scene, tiles: new Map(), blobs: new Map(), busy: 0, key: '', geo: quad(), aniso: Math.min(8, renderer.capabilities.getMaxAnisotropy()), renderer, born: performance.now() });
    fetch('data/ground.json').then(r => r.json()).then(m => {
      const is = m.tiles.map(t => t[0]), js = m.tiles.map(t => t[1]), s = m.size;
      this.i0 = Math.min(...is) - 1; this.j0 = Math.min(...js) - 1;
      this.ni = Math.max(...is) - this.i0 + 2; this.nj = Math.max(...js) - this.j0 + 2;
      const tex = new THREE.DataTexture(new Uint8Array(this.ni * this.nj), this.ni, this.nj, THREE.RedFormat);
      tex.unpackAlignment = 1;
      looks.cover.value = this.cover = tex;
      looks.cbox.value.set(this.i0 * s, this.j0 * s, 1 / (this.ni * s), 1 / (this.nj * s));
      this.have = new Set(m.tiles.map(t => t.join('_')));
      this.pre = m.tiles.map(([i, j]) => [Math.hypot((i + 0.5) * s - HYPE[0], (j + 0.5) * s - HYPE[1]), i, j]).sort((a, b) => a[0] - b[0]).slice(0, 80);
      this.meta = m;
    }).catch(() => {});
  }

  tile(k, i, j) {
    return this.tiles.get(k) || this.tiles.set(k, { i, j, a: 0, res: 0, on: false, used: 0, show: { value: 0 } }).get(k);
  }

  update(camera, dt) {
    const m = this.meta;
    if (!m) return;
    const s = m.size, near = looks.near.value;
    if (near) {
      const res = [160, 320, 640, 1280].find(r => r >= s * camera.zoom * this.renderer.getPixelRatio() * 1.3) || 1280;
      const key = stamp(camera, res, this.busy);
      if (key !== this.key) { this.key = key; this.plan(camera, res); }
    } else if (!this.busy && this.pre.length && performance.now() - this.born > 4000) {
      const [, i, j] = this.pre.shift(), k = `${i}_${j}`, t = this.tile(k, i, j);
      if (!t.res) this.load(k, t, 160);
    }
    const d = this.cover.image.data;
    let dirty = false;
    for (const t of this.tiles.values()) {
      t.a = t.mesh ? Math.min(t.a + dt / 0.2, 1) : 0;
      const val = t.on ? t.a * near : 0, o = (t.j - this.j0) * this.ni + t.i - this.i0, b = Math.round(val * 255);
      if (t.show.value !== val) t.show.value = val;
      if (t.mesh) t.mesh.visible = val > 0;
      if (d[o] !== b) { d[o] = b; dirty = true; }
    }
    if (dirty) this.cover.needsUpdate = true;
  }

  plan(camera, res) {
    const s = this.meta.size, r = 2 * camera.zoom / innerWidth * s, want = [], now = performance.now();
    for (const k of this.have) {
      const [i, j] = k.split('_').map(Number);
      v.set((i + 0.5) * s, 0, -(j + 0.5) * s).project(camera);
      if (Math.abs(v.x) < 1 + r && Math.abs(v.y) < 1 + r) want.push([v.x * v.x + v.y * v.y, k, i, j]);
    }
    want.sort((a, b) => a[0] - b[0]);
    want.length = Math.min(want.length, caps.ground, Math.floor(caps.texels / res / res));
    for (const t of this.tiles.values()) t.on = false;
    const ts = want.map(([, k, i, j]) => Object.assign(this.tile(k, i, j), { on: true, used: now, k }));
    for (const t of [...ts.filter(t => !t.res), ...ts.filter(t => t.res && t.res < res)]) {
      if (this.busy >= caps.loads) break;
      if (!t.busy) this.load(t.k, t, t.res ? res : 160);
    }
    let sum = 0;
    for (const t of this.tiles.values()) sum += t.res * t.res;
    for (const [k, t] of [...this.tiles].filter(([, t]) => !t.on && t.mesh).sort((a, b) => a[1].used - b[1].used)) {
      if (sum <= caps.cache) break;
      sum -= t.res * t.res;
      this.drop(k);
    }
  }

  load(k, t, res) {
    t.busy = true; this.busy++;
    const blob = this.blobs.get(k) || fetch(`data/ground/${k}.webp`).then(r => r.blob());
    this.blobs.set(k, blob);
    blob.then(b => createImageBitmap(b, { resizeWidth: res, resizeHeight: res, resizeQuality: 'high' })).then(img => {
      if (this.tiles.get(k) !== t || t.res >= res) return img.close();
      const tex = new THREE.Texture(img), s = this.meta.size;
      Object.assign(tex, { colorSpace: THREE.SRGBColorSpace, flipY: false, anisotropy: this.aniso, needsUpdate: true, onUpdate: () => img.close() });
      if (t.mesh) { t.mesh.material.map.dispose(); t.mesh.material.map = tex; }
      else {
        t.mesh = new THREE.Mesh(this.geo, patch(new THREE.MeshStandardMaterial({ roughness: 1, map: tex, alphaTest: 0.5, depthTest: false, depthWrite: false }), '', 'diffuseColor.rgb *= uDim;', '', looks.one, t.show));
        t.mesh.position.set(t.i * s, 0, -t.j * s);
        t.mesh.scale.set(s, 1, s);
        t.mesh.renderOrder = -0.5;
        t.mesh.receiveShadow = true;
        t.mesh.visible = false;
        this.scene.add(t.mesh);
      }
      t.res = res;
    }).catch(() => { this.blobs.delete(k); }).finally(() => { t.busy = false; this.busy--; this.key = ''; });
  }

  drop(k) {
    const t = this.tiles.get(k);
    this.tiles.delete(k);
    if (this.cover) this.cover.image.data[(t.j - this.j0) * this.ni + t.i - this.i0] = 0, this.cover.needsUpdate = true;
    if (!t.mesh) return;
    this.scene.remove(t.mesh);
    t.mesh.material.map.dispose();
    t.mesh.material.dispose();
  }
}

const SPRITE_VERT = `attribute vec4 aI; uniform vec4 uRect[NS]; uniform vec4 uInfo[NS]; varying vec2 vUv;
  void main() {
    int k = int(aI.z); vec4 r = uRect[k], f = uInfo[k]; float sc = abs(aI.w), flip = step(aI.w, 0.0), bx = mix(f.x, 1.0 - f.x, flip);
    vec3 right = normalize(vec3(viewMatrix[0][0], 0.0, viewMatrix[2][0]));
    vec3 p = vec3(aI.x, 0.3, -aI.y) + right * (position.x - bx) * f.z * sc + vec3(0.0, position.y * f.y * sc, 0.0);
    vUv = vec2(mix(r.x, r.z, mix(position.x, 1.0 - position.x, flip)), mix(r.y, r.w, position.y));
    gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
  }`;
const SPRITE_FRAG = `uniform sampler2D uMap; uniform float uDim, uSat, uShow; varying vec2 vUv;
  void main() {
    vec4 c = texture2D(uMap, vUv);
    if (c.a < 0.45 || ${DITHER} >= uShow) discard;
    c.rgb = mix(vec3(dot(c.rgb, vec3(0.3, 0.59, 0.11))), c.rgb, uSat) * uDim * 1.1;
    gl_FragColor = vec4(c.rgb, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;
const BLOB_VERT = `attribute vec4 aI; uniform vec4 uInfo[NS]; varying vec2 vQ;
  void main() {
    vec4 f = uInfo[int(aI.z)]; float sc = abs(aI.w), h = f.y * sc, w = f.z * sc * 0.4;
    vec2 d = vec2(${SHADOW}), n = vec2(-d.y, d.x);
    vQ = position.xy * 2.0 - 1.0;
    vec2 c = aI.xy + d * (0.12 * h + vQ.x * (w + 0.1 * h)) + n * vQ.y * w * 0.85;
    gl_Position = projectionMatrix * viewMatrix * vec4(c.x, 0.56, -c.y, 1.0);
  }`;
const BLOB_FRAG = `uniform float uShow, uDim; varying vec2 vQ;
  void main() { float r = dot(vQ, vQ); if (r > 1.0) discard; gl_FragColor = vec4(0.0, 0.0, 0.0, 0.55 * smoothstep(1.0, 0.2, r) * uShow * uDim); }`;

class Sprites {
  constructor(scene) {
    this.scene = scene;
    this.ready = fetch('assets/sprites/sprites.json').then(r => r.json()).then(list => Promise.all(list.map(s => new Promise((ok, no) => {
      const img = new Image();
      img.onload = () => ok(Object.assign(s, { img }));
      img.onerror = no;
      img.src = `assets/sprites/${s.name}.png`;
    })))).then(list => this.build(list)).catch(() => {});
  }

  build(list) {
    const W = 2048, pad = 8;
    let x = 0, y = 0, row = 0;
    for (const s of list) {
      if (x + s.w > W) { x = 0; y += row + pad; row = 0; }
      Object.assign(s, { x, y });
      x += s.w + pad; row = Math.max(row, s.h);
    }
    const H = y + row, cv = Object.assign(document.createElement('canvas'), { width: W, height: H }), cx = cv.getContext('2d');
    for (const s of list) cx.drawImage(s.img, s.x, s.y, s.w, s.h);
    const map = new THREE.CanvasTexture(cv);
    Object.assign(map, { colorSpace: THREE.SRGBColorSpace, anisotropy: 4 });
    this.list = list;
    this.groups = GROUPS.map(g => g.map(p => list.flatMap((s, k) => s.name.startsWith(p) ? [k] : [])));
    this.walkers = list.flatMap((s, k) => s.name.startsWith('walker') ? [k] : []);
    const uni = { uMap: { value: map }, uDim: looks.dim, uSat: looks.sat, uShow: looks.near,
      uRect: { value: list.map(s => new THREE.Vector4(s.x / W, 1 - (s.y + s.h) / H, (s.x + s.w) / W, 1 - s.y / H)) },
      uInfo: { value: list.map(s => new THREE.Vector4(s.base[0], s.m, s.m * s.w / s.h, 0)) } };
    const defines = { NS: list.length };
    const sprite = new THREE.ShaderMaterial({ uniforms: uni, defines, vertexShader: SPRITE_VERT, fragmentShader: SPRITE_FRAG });
    const blob = new THREE.ShaderMaterial({ uniforms: uni, defines, vertexShader: BLOB_VERT, fragmentShader: BLOB_FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    const base = new THREE.PlaneGeometry(1, 1).translate(0.5, 0.5, 0);
    this.sets = [caps.sprites, caps.walkers].map(n => {
      const g = new THREE.InstancedBufferGeometry().copy(base), a = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage);
      g.setAttribute('aI', a);
      g.instanceCount = 0;
      const ms = [new THREE.Mesh(g, sprite), new THREE.Mesh(g, blob)];
      ms[1].renderOrder = 5;
      for (const m of ms) { m.frustumCulled = false; this.scene.add(m); }
      return { g, a, ms };
    });
  }

  prep(t, g, kinds) {
    if (!this.groups) return;
    const tr = g.userData.trees, out = new Float32Array(tr.length / 3 * 4);
    for (let k = 0, o = 0; k < tr.length; k += 3, o += 4) {
      const gr = this.groups[tr[k + 2]], vs = gr[Math.floor(Math.random() * gr.length)];
      out.set([tr[k], tr[k + 1], vs[Math.floor(Math.random() * vs.length)], (0.8 + 0.4 * Math.random()) * (Math.random() < 0.5 ? -1 : 1)], o);
    }
    g.userData.spr = out;
    const paths = (t.r || []).filter(([q]) => kinds.r[q] === 'footway').map(([, , f]) => f)
      .concat((t.g || []).filter(([q]) => kinds.g[q] === 'sidewalk').map(([, f]) => f.concat(f.slice(0, 2))))
      .map(f => { const p = f.reduce((a, v, i) => (i % 2 ? a[a.length - 1].push(v) : a.push([v]), a), []), xs = p.map(q => q[0]), ys = p.map(q => q[1]); return { g: p, cum: lengths(p), b: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] }; })
      .filter(p => p.cum[p.cum.length - 1] > 5);
    g.userData.walk = paths;
  }

  trees(tiles, rect) {
    const [x0, y0, x1, y1] = rect, A = this.sets[0].a.array;
    let total = 0, n = 0, acc = 0;
    for (const t of tiles) for (let k = 0; k < t.length; k += 4) if (t[k] > x0 && t[k] < x1 && t[k + 1] > y0 && t[k + 1] < y1) total++;
    const step = Math.max(total / caps.sprites, 1) / Math.max(looks.trees.value, 1e-3);
    for (const t of tiles) for (let k = 0; k < t.length; k += 4) {
      if (!(t[k] > x0 && t[k] < x1 && t[k + 1] > y0 && t[k + 1] < y1) || (acc += 1) < step) continue;
      acc -= step;
      if (n < caps.sprites) A.set(t.subarray(k, k + 4), 4 * n++);
    }
    this.sets[0].g.instanceCount = n;
    this.sets[0].a.clearUpdateRanges();
    this.sets[0].a.addUpdateRange(0, n * 4);
    this.sets[0].a.needsUpdate = true;
  }
}

class Walkers {
  constructor() {
    const N = caps.walkers;
    Object.assign(this, { q: new Array(N), s: new Float32Array(N), d: new Float32Array(N), id: new Float32Array(N), paths: [], cdf: [], n: 0, gen: 0, density: 1, e: [0, 0, 0, 0] });
  }

  plan(tiles, rect) {
    const [x0, y0, x1, y1] = rect;
    this.gen++;
    this.paths = [];
    let s = 0;
    for (const t of tiles) for (const p of t) if (p.b[0] < x1 && p.b[2] > x0 && p.b[1] < y1 && p.b[3] > y0) { p.gen = this.gen; this.paths.push(p); }
    this.cdf = this.paths.map(p => s += p.cum[p.cum.length - 1]);
    this.n = Math.min(caps.walkers, Math.floor(s / 20 * this.density));
  }

  tick(dt, set, ids, rx, rz) {
    const A = set.a.array, e = this.e;
    for (let p = 0; p < this.n; p++) {
      let q = this.q[p];
      if (!q || q.gen !== this.gen) {
        const r = Math.random() * this.cdf[this.cdf.length - 1];
        let lo = 0, hi = this.cdf.length - 1;
        while (lo < hi) { const m = (lo + hi) >> 1; if (this.cdf[m] < r) lo = m + 1; else hi = m; }
        q = this.q[p] = this.paths[lo];
        this.s[p] = Math.random() * q.cum[q.cum.length - 1];
        this.d[p] = Math.random() < 0.5 ? -1 : 1;
        this.id[p] = ids[p % ids.length];
      }
      const L = q.cum[q.cum.length - 1];
      let s = this.s[p] + this.d[p] * 1.3 * dt;
      if (s < 0 || s > L) { this.d[p] *= -1; s = Math.min(Math.max(s, 0), L); }
      this.s[p] = s;
      along(q.g, q.cum, s, e);
      const side = (p % 2 ? 1 : -1) * 0.6;
      A[4 * p] = e[0] + e[3] * side; A[4 * p + 1] = e[1] - e[2] * side; A[4 * p + 2] = this.id[p];
      A[4 * p + 3] = (e[2] * rx - e[3] * rz) * this.d[p] > 0 ? -1 : 1;
    }
    set.g.instanceCount = this.n;
    set.a.needsUpdate = true;
  }
}

function props() {
  const paint = (g, c) => { const t = new THREE.Color(c); g = g.index ? g.toNonIndexed() : g; g.deleteAttribute('uv'); g.setAttribute('color', new THREE.Float32BufferAttribute(Array(g.attributes.position.count).fill([t.r, t.g, t.b]).flat(), 3)); return g; };
  const box = (w, h, d, x, y, z, c) => paint(new THREE.BoxGeometry(w, h, d).translate(x, y + h / 2, z), c);
  const pole = (h, r, c) => paint(new THREE.CylinderGeometry(r * 0.8, r, h, 6).translate(0, h / 2, 0), c);
  const POLE = '#c4c6c7', GLASS = '#c3d6dd';
  return {
    lamp: [pole(7, 0.08, POLE), box(0.2, 0.08, 1.2, 0, 6.95, 0.45, POLE), box(0.3, 0.12, 0.55, 0, 6.85, 0.9, '#f1eee4')],
    signal: [pole(3.4, 0.07, POLE), box(0.3, 0.95, 0.28, 0, 2.5, 0.12, '#55585b')],
    bus_stop: [box(4.2, 0.08, 1.7, 0, 2.4, 0, GLASS), box(4, 1.9, 0.04, 0, 0.35, -0.75, GLASS), box(1.8, 0.06, 0.4, 0, 0.45, -0.5, POLE),
      ...[[-1.95, -0.75], [1.95, -0.75], [-1.95, 0.75], [1.95, 0.75]].map(([x, z]) => box(0.07, 2.4, 0.07, x, 0, z, POLE))],
    bench: [box(1.8, 0.06, 0.45, 0, 0.42, 0, '#cbbfae'), box(1.8, 0.4, 0.05, 0, 0.5, -0.22, '#cbbfae'), box(0.06, 0.42, 0.45, -0.8, 0, 0, POLE), box(0.06, 0.42, 0.45, 0.8, 0, 0, POLE)]
  };
}

export class Near {
  constructor(scene, renderer, city) {
    this.city = city;
    this.ground = new Ground(scene, renderer);
    this.sprites = new Sprites(scene);
    this.walkers = new Walkers();
    this.o = new THREE.Object3D();
    const mat = patch(new THREE.MeshStandardMaterial({ roughness: 0.8, vertexColors: true }), '', 'diffuseColor.rgb *= uDim;', '', looks.one, looks.near);
    this.props = Object.entries(props()).map(([name, parts]) => {
      const m = new THREE.InstancedMesh(mergeGeometries(parts), mat, caps.props);
      m.count = 0;
      m.castShadow = m.receiveShadow = true;
      m.frustumCulled = false;
      scene.add(m);
      return { name, m };
    });
    city.onTile = (t, g) => {
      const k = city.meta, P = {};
      for (const [q, x, y, a] of t.p || []) (P[k.pkinds[q]] ||= []).push(x, y, a);
      g.userData.props = P;
      this.sprites.prep(t, g, { r: k.rkinds, g: k.gkinds });
    };
    this.key = '';
  }

  update(camera, rect, dt, mode) {
    this.ground.update(camera, dt);
    const on = looks.near.value > 0 && mode !== 'flow', sets = this.sprites.sets;
    for (const { m } of this.props) m.visible = on;
    for (const s of sets || []) for (const m of s.ms) m.visible = on;
    if (!on || !sets) return;
    const key = stamp(camera, this.city.loaded.size, this.city.busy);
    if (key !== this.key) {
      this.key = key;
      const [x0, y0, x1, y1] = rect, d = 40, r = [x0 - d, y0 - d, x1 + d, y1 + d];
      const tiles = [...this.city.loaded.values()].filter(g => g?.userData.spr && g.name.split('_').every((v, i) => v * 1000 < r[i + 2] && v * 1000 + 1000 > r[i]));
      this.sprites.trees(tiles.map(g => g.userData.spr), r);
      this.walkers.plan(tiles.map(g => g.userData.walk), r);
      for (const { name, m } of this.props) {
        let n = 0;
        for (const g of tiles) {
          const a = g.userData.props[name] || [];
          for (let k = 0; k < a.length && n < caps.props; k += 3) {
            if (a[k] < r[0] || a[k] > r[2] || a[k + 1] < r[1] || a[k + 1] > r[3]) continue;
            this.o.position.set(a[k], 0, -a[k + 1]);
            this.o.rotation.y = a[k + 2];
            this.o.updateMatrix();
            m.setMatrixAt(n++, this.o.matrix);
          }
        }
        m.count = n;
        m.instanceMatrix.needsUpdate = true;
      }
    }
    const e = camera.matrixWorld.elements;
    this.walkers.tick(dt, sets[1], this.sprites.walkers, e[0], e[2]);
  }
}
