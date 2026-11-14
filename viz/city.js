import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export const HYPE = [-1511.7, 1239.2];
export const looks = { dim: { value: 1 }, sat: { value: 1 }, tree: { value: 1 }, lane: { value: 1 }, path: { value: 1 }, one: { value: 1 }, px: { value: 1 }, coarse: { value: 0 }, fine: { value: 1 }, near: { value: 0 }, far: { value: 1 }, out: { value: 0.3 }, mark: { value: new THREE.Vector3(1, 1, 1) }, trees: { value: 1 }, sky: { value: new THREE.Vector3() },
  cover: { value: new THREE.DataTexture(new Uint8Array(4), 1, 1) }, cbox: { value: new THREE.Vector4() } };

const GROUND = { land: '#8e8f8b', water: '#15191c', forest: '#4f6339', park: '#667f48', grass: '#6f864e', field: '#959872',
  wetland: '#6d7a58', sand: '#b5ad92', rock: '#97958f', paved: '#9a9b97', rail: '#7d7a74', pitch: '#7a9c56',
  parking: '#8b8c88', playground: '#bfa98a', scrub: '#5f7342', garden: '#6b8449', hedge: '#3e5530', sidewalk: '#a9aaa6', conifer_forest: '#3d5236' };
const BASE = new THREE.Color(GROUND.water);

const hash = `float hash(float n){return fract(sin(n*12.9898)*43758.5453);}`;
export const DITHER = 'fract(dot(gl_FragCoord.xy, vec2(0.7548777, 0.5698403)))';
export const COVER = 'uniform sampler2D uCover; uniform vec4 uCbox; float cover(vec2 w) { return texture2D(uCover, (vec2(w.x, -w.y) - uCbox.xy) * uCbox.zw).r; }';

export function fadeDepth(show) {
  const m = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
  m.customProgramCacheKey = () => 'fade';
  m.onBeforeCompile = s => {
    s.uniforms.uShow = show;
    s.fragmentShader = 'uniform float uShow;\n' + s.fragmentShader.replace('#include <alphatest_fragment>', `#include <alphatest_fragment>\nif (${DITHER} >= uShow) discard;`);
  };
  return m;
}

export function patch(mat, vert = '', frag = '', pars = '', fade = looks.one, show = looks.one) {
  mat.customProgramCacheKey = () => vert + frag + pars;
  mat.onBeforeCompile = s => {
    Object.assign(s.uniforms, { uDim: looks.dim, uSat: looks.sat, uTree: looks.tree, uLane: looks.lane, uFade: fade, uPx: looks.px, uShow: show, uSky: looks.sky, uCover: looks.cover, uCbox: looks.cbox });
    s.vertexShader = s.vertexShader.replace('#include <common>', `#include <common>\nuniform float uTree;\n${pars}`).replace('#include <begin_vertex>', `#include <begin_vertex>\n${vert}`);
    s.fragmentShader = s.fragmentShader.replace('#include <common>', `#include <common>\nuniform float uDim, uSat, uLane, uFade, uShow;\nuniform vec3 uSky;\n${hash}\n${COVER}\n${pars.replace(/attribute[^;]*;/g, '')}`)
      .replace('#include <dithering_fragment>', '#include <dithering_fragment>\nif (sky > 0.5) gl_FragColor.rgb = uSky;')
      .replace('#include <color_fragment>', `#include <color_fragment>\nfloat sky = 0.0;\nif (${DITHER} >= uShow) discard;\n${frag}\ndiffuseColor.rgb = mix(vec3(dot(diffuseColor.rgb, vec3(0.3, 0.59, 0.11))), diffuseColor.rgb, uSat);`);
  };
  return mat;
}

const buildingLook = show => patch(new THREE.MeshStandardMaterial({ roughness: 0.92 }), 'vB = aB;', `
  float k = mod(vB.z, 100.0), roof = step(99.5, vB.z), r = hash(k + 3.0);
  vec3 c = mix(vec3(0.6, 0.61, 0.635), vec3(0.54, 0.55, 0.57), roof) * (0.93 + 0.1 * hash(k));
  float fw = 2.6 + 1.6 * r, fh = 3.1 + 0.5 * hash(k + 7.0);
  vec2 g = fract(vec2(vB.x / fw, vB.y / fh));
  float win = step(0.22, g.x) * step(g.x, 0.78) * step(0.32, g.y) * step(g.y, 0.8) * step(2.0, vB.y);
  win = mix(0.27, win, clamp(1.6 - 4.0 * max(fwidth(vB.x / fw), fwidth(vB.y / fh)), 0.0, 1.0));
  c *= 1.0 - (1.0 - roof) * (0.2 * win + 0.4 * (1.0 - smoothstep(0.0, 6.0, vB.y)));
  c *= 1.0 - roof * 0.05 * mix(0.5, step(0.5, fract(vB.x * 0.25)), clamp(1.6 - 4.0 * fwidth(vB.x * 0.25), 0.0, 1.0));
  diffuseColor.rgb = c * uDim;`, 'attribute vec3 aB; varying vec3 vB;', looks.one, show);
const buildingMat = buildingLook(looks.one), coarseMat = buildingLook(looks.coarse), coarseDepth = fadeDepth(looks.coarse);
const dress = (m, mat) => { m.material = mat; m.customDepthMaterial = mat === coarseMat ? coarseDepth : undefined; };

const roadLook = fade => patch(new THREE.MeshStandardMaterial({ roughness: 0.9, vertexColors: true }), 'vB = aB; vW = (modelMatrix * vec4(position, 1.0)).xz;', `
  float n = max(vB.z, 1.0), t = (vB.y * 0.5 + 0.5) * n;
  float focus = step(length(vW - vec2(${HYPE[0].toFixed(1)}, ${(-HYPE[1]).toFixed(1)})), 1500.0) * step(1.5, vB.z);
  float lane = step(abs(fract(t + 0.5) - 0.5), 0.035) * step(0.5, t) * step(t, n - 0.5) * step(0.5, fract(vB.x / 9.0));
  float edge = step(0.9, abs(vB.y)) * step(abs(vB.y), 0.95);
  if (${DITHER} < cover(vW)) discard;
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.92, 0.92, 0.88), focus * uLane * max(lane, edge));
  diffuseColor.rgb = mix(vec3(0.27, 0.275, 0.26), diffuseColor.rgb, uFade) * uDim;`, 'attribute vec3 aB; varying vec3 vB; varying vec2 vW;', fade);
const roadMat = roadLook(looks.one), pathMat = roadLook(looks.path);
const lineMat = (c, a) => new THREE.ShaderMaterial({ uniforms: { uPx: looks.px, uA: { value: a } },
  vertexShader: 'attribute vec3 aE; uniform float uPx; void main() { vec3 p = position; p.xz += aE.xy * aE.z * uPx * 0.5; gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); }',
  fragmentShader: `uniform float uA; void main() { gl_FragColor = vec4(vec3(${new THREE.Color(c).toArray().map(v => v.toFixed(3))}) * uA, 1.0); }` });
export const borderMat = lineMat('#8a8a85', 1);
const overMat = patch(new THREE.MeshStandardMaterial({ roughness: 0.9, vertexColors: true, depthTest: false, depthWrite: false }),
  'transformed.xz += aE.xy * max(0.0, uPx * 2.5 - aE.z) * 0.5;', 'diffuseColor.rgb *= uDim;', 'attribute vec3 aE; uniform float uPx;');

const groundMat = patch(new THREE.MeshStandardMaterial({ roughness: 1, vertexColors: true, depthWrite: false, depthTest: false }), '',
  `diffuseColor.rgb *= uDim; sky = step(distance(vColor, vec3(${BASE.toArray().map(v => v.toFixed(5))})), 0.001);`);
const treeMat = patch(new THREE.MeshStandardMaterial({ roughness: 1, vertexColors: true }), 'transformed *= uTree;', 'diffuseColor.rgb *= uDim;', '', looks.one, looks.far);
const SCATTER = { forest: [3, 140], conifer_forest: [4, 140], hedge: [5, 6], scrub: [5, 40], garden: [5, 90] };

const treeGeo = (() => {
  const crown = new THREE.IcosahedronGeometry(1, 0).scale(2.5, 3.2, 2.5).translate(0, 5, 0);
  const trunk = new THREE.CylinderGeometry(0.22, 0.3, 3, 4, 1, true).translate(0, 1.5, 0);
  const paint = (g, c) => g.setAttribute('color', new THREE.Float32BufferAttribute(Array(g.attributes.position.count).fill(c).flat(), 3));
  paint(crown, [1, 1, 1]); paint(trunk, [0.5, 0.42, 0.35]);
  return mergeGeometries([crown, trunk.toNonIndexed()]);
})();

function ring(f) {
  const p = [];
  for (let i = 0; i < f.length; i += 2) p.push(new THREE.Vector2(f[i], f[i + 1]));
  if (p.length > 2 && p[0].equals(p[p.length - 1])) p.pop();
  if (THREE.ShapeUtils.isClockWise(p)) p.reverse();
  return p;
}

function tris(p) {
  return THREE.ShapeUtils.triangulateShape(p, []).map(([a, b, c]) =>
    (p[b].x - p[a].x) * (p[c].y - p[a].y) - (p[b].y - p[a].y) * (p[c].x - p[a].x) < 0 ? [a, c, b] : [a, b, c]);
}

export function up(g) {
  const n = new Float32Array(g.attributes.position.count * 3);
  for (let i = 1; i < n.length; i += 3) n[i] = 1;
  g.setAttribute('normal', new THREE.BufferAttribute(n, 3));
  return g;
}

function inside(p, x, y) {
  let c = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++)
    if ((p[i].y > y) !== (p[j].y > y) && x < (p[j].x - p[i].x) * (y - p[i].y) / (p[j].y - p[i].y) + p[i].x) c = !c;
  return c;
}

function buildings(list, mat = buildingMat) {
  const pos = [], nor = [], ab = [];
  for (const [h, kind, p] of list) {
    let u = 0;
    for (let i = 0; i < p.length; i++) {
      const a = p[i], b = p[(i + 1) % p.length], l = a.distanceTo(b);
      if (l < 0.01) continue;
      const nx = (b.y - a.y) / l, nz = (b.x - a.x) / l;
      for (const [q, y, s] of [[a, 0, 0], [b, 0, l], [b, h, l], [a, 0, 0], [b, h, l], [a, h, 0]]) {
        pos.push(q.x, y, -q.y); nor.push(nx, 0, nz); ab.push(u + s, y, kind);
      }
      u += l;
    }
    for (const t of tris(p))
      for (const i of t) { pos.push(p[i].x, h, -p[i].y); nor.push(0, 1, 0); ab.push(p[i].x, p[i].y, kind + 100); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('aB', new THREE.Float32BufferAttribute(ab, 3));
  const m = new THREE.Mesh(g);
  dress(m, mat);
  m.castShadow = m.receiveShadow = true;
  return m;
}

function ground(list, kinds, size, i, j, trees) {
  const pos = [], col = [], c = new THREE.Color();
  const quad = (x0, y0, x1, y1, h, cc) => { for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]) { pos.push(x, h, -y); col.push(cc.r, cc.g, cc.b); } };
  quad(i * size, j * size, (i + 1) * size, (j + 1) * size, -0.05, BASE);
  const polys = list.map(([k, f]) => {
    const p = ring(f), b = [Infinity, Infinity, -Infinity, -Infinity];
    for (const v of p) { b[0] = Math.min(b[0], v.x); b[1] = Math.min(b[1], v.y); b[2] = Math.max(b[2], v.x); b[3] = Math.max(b[3], v.y); }
    return { name: kinds[k] || '', p, b };
  }).filter(q => q.p.length > 2);
  const covered = (from, x, y) => polys.some((q, k) => k > from && !SCATTER[q.name] && x > q.b[0] && x < q.b[2] && y > q.b[1] && y < q.b[3] && inside(q.p, x, y));
  polys.forEach(({ name, p, b }, k) => {
    c.set(GROUND[name] || GROUND.land);
    for (const t of tris(p))
      for (const v of t) { pos.push(p[v].x, 0, -p[v].y); col.push(c.r, c.g, c.b); }
    const sc = SCATTER[name];
    if (!sc || !trees) return;
    const n = Math.min(3000, (b[2] - b[0]) * (b[3] - b[1]) / sc[1]);
    for (let s = 0; s < n && trees.length < 48000; s++) {
      const x = b[0] + Math.random() * (b[2] - b[0]), y = b[1] + Math.random() * (b[3] - b[1]);
      if (inside(p, x, y) && !covered(k, x, y)) trees.push(x, y, sc[0]);
    }
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const m = new THREE.Mesh(up(g), groundMat);
  m.receiveShadow = true;
  m.renderOrder = -1;
  return m;
}

const roadStyle = k => /rail|tram|subway|metro/.test(k) ? [2.4, '#55524d', 0.48] : /foot|path|cycle|steps|pedestrian|track|bridleway/.test(k) ? [1.2, '#979893', 0.4]
  : /service|living/.test(k) ? [3.5, '#7a7c7d', 0.42] : [0, '#5e6164', 0.45];

export function ribbon(lines, extra) {
  const pos = [], ab = [], col = [], idx = [], c = new THREE.Color();
  for (const { pts, w, color, y, lanes, ...rest } of lines) {
    c.set(color || '#000');
    const n = pts.length / 2, base = pos.length / 3;
    let u = 0;
    for (let k = 0; k < n; k++) {
      const x = pts[2 * k], z = pts[2 * k + 1];
      const pk = Math.max(k - 1, 0), nk = Math.min(k + 1, n - 1);
      let dx = pts[2 * nk] - pts[2 * pk], dz = pts[2 * nk + 1] - pts[2 * pk + 1];
      const dl = Math.hypot(dx, dz) || 1; dx /= dl; dz /= dl;
      if (k) u += Math.hypot(x - pts[2 * k - 2], z - pts[2 * k - 1]);
      for (const s of [-1, 1]) {
        pos.push(x - dz * s * w / 2, y, -(z + dx * s * w / 2)); ab.push(u, s, lanes); col.push(c.r, c.g, c.b);
        extra && extra(rest, s, dx, dz);
      }
      if (k) idx.push(base + 2 * k - 2, base + 2 * k, base + 2 * k - 1, base + 2 * k - 1, base + 2 * k, base + 2 * k + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aB', new THREE.Float32BufferAttribute(ab, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  return g;
}

function roads(list, kinds, minor) {
  const lines = list.filter(([k]) => kinds[k] !== 'tunnel' && minor === /foot|cycle|service|track/.test(kinds[k])).map(([k, lanes, f]) => {
    const [w, color, y] = roadStyle(kinds[k] || '');
    return { pts: f, w: w || Math.max(lanes, 1) * 3.3 + 1, color, y, lanes: w ? 0 : lanes };
  });
  const m = new THREE.Mesh(up(ribbon(lines)), minor ? pathMat : roadMat);
  m.receiveShadow = true;
  return m;
}

function forest(all) {
  const xy = all.filter((v, i) => all[i - i % 3 + 2] < 5), n = xy.length / 3, m = new THREE.InstancedMesh(treeGeo, treeMat, n), o = new THREE.Object3D(), c = new THREE.Color();
  for (let k = 0; k < n; k++) {
    o.position.set(xy[3 * k], 0, -xy[3 * k + 1]);
    o.rotation.y = Math.random() * 6.28;
    o.scale.setScalar(0.7 + Math.random() * 0.7);
    o.updateMatrix(); m.setMatrixAt(k, o.matrix);
    m.setColorAt(k, c.setHSL(0.22 + Math.random() * 0.08, 0.35 + Math.random() * 0.15, 0.27 + Math.random() * 0.1));
  }
  m.castShadow = true;
  return m;
}

export class City {
  constructor(scene) {
    this.root = new THREE.Group();
    scene.add(this.root);
    this.loaded = new Map();
    this.hidden = [];
    this.lod = 0;
    this.busy = 0;
    this.whole = new Set();
    this.ready = fetch('data/tiles.json').then(r => r.json()).then(m => { this.meta = m; }).catch(() => {});
    Promise.all([this.ready, fetch('data/overview.json').then(r => r.json())]).then(([, o]) => {
      const e = [], lines = (o.r || []).filter(([k]) => /motorway|trunk|primary|secondary/.test(this.meta.rkinds[k])).map(([, lanes, f]) => ({ pts: f, w: Math.max(lanes, 1) * 3.3 + 1, ww: Math.max(lanes, 1) * 3.3 + 1, color: '#5e6164', y: 0 }));
      const g = ground(o.g || [], this.meta.gkinds, 0, 0, 0, null), r = new THREE.Mesh(up(ribbon(lines, ({ ww }, s, dx, dz) => e.push(-dz * s, -dx * s, ww))), overMat);
      r.geometry.setAttribute('aE', new THREE.Float32BufferAttribute(e, 3));
      g.renderOrder = -3; r.renderOrder = -2;
      scene.add(g, r);
      const round = f => { for (let n = 4; n--;) { const g = []; for (let i = 0; i < f.length; i += 2) { const j = (i + 2) % f.length; g.push(0.75 * f[i] + 0.25 * f[j], 0.75 * f[i + 1] + 0.25 * f[j + 1], 0.25 * f[i] + 0.75 * f[j], 0.25 * f[i + 1] + 0.75 * f[j + 1]); } f = g; } return [...f, f[0], f[1]]; };
      const rings = [(o.border || []).map(round)];
      rings[0].forEach(f => {
        const q = [], m = new THREE.Mesh(ribbon([{ pts: f, w: 0, y: 0.6, px: 2 }], ({ px }, s, dx, dz) => q.push(-dz * s, -dx * s, px)), borderMat);
        m.geometry.setAttribute('aE', new THREE.Float32BufferAttribute(q, 3));
        m.frustumCulled = false;
        scene.add(m);
      });
      if (rings[0].length) {
        const all = rings[0].flat(), xs = all.filter((v, i) => !(i % 2)), ys = all.filter((v, i) => i % 2), d = 3000, S = 2048;
        const x0 = Math.min(...xs) - d, y0 = Math.min(...ys) - d, w = Math.max(...xs) + d - x0, h = Math.max(...ys) + d - y0;
        const far = 4e5, shape = new THREE.Shape([[-far, -far], [far, -far], [far, far], [-far, far]].map(p => new THREE.Vector2(...p)));
        for (const f of rings[0]) { const p = []; for (let i = 0; i < f.length - 2; i += 2) p.push(new THREE.Vector2(f[i], f[i + 1])); shape.holes.push(new THREE.Path(p)); }
        const shade = new THREE.Mesh(new THREE.ShapeGeometry(shape).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({ uniforms: { uOut: looks.out }, depthTest: false, depthWrite: false, transparent: true,
          blending: THREE.CustomBlending, blendSrc: THREE.ZeroFactor, blendDst: THREE.SrcColorFactor,
          vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }', fragmentShader: 'uniform float uOut; void main() { gl_FragColor = vec4(vec3(pow(uOut, 0.4545)), 1.0); }' }));
        shade.renderOrder = 50; shade.frustumCulled = false;
        scene.add(shade);
        const f = rings[0].reduce((a, b) => b.length > a.length ? b : a);
        let A = 0, mx = 0, my = 0;
        for (let i = 0; i < f.length - 2; i += 2) { const c = f[i] * f[i + 3] - f[i + 2] * f[i + 1]; A += c; mx += (f[i] + f[i + 2]) * c; my += (f[i + 1] + f[i + 3]) * c; }
        mx /= 3 * A; my /= 3 * A;
        this.tag = [mx, my + 0.3 * (Math.max(...ys) - my)];
      }
    }).catch(() => {});
  }

  update(x0, y0, x1, y1) {
    const m = this.meta;
    if (!m) return;
    const s = m.size, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const want = m.tiles.filter(([i, j]) => i * s < x1 && (i + 1) * s > x0 && j * s < y1 && (j + 1) * s > y0)
      .sort((a, b) => Math.hypot(a[0] * s - cx, a[1] * s - cy) - Math.hypot(b[0] * s - cx, b[1] * s - cy)).slice(0, 300);
    const keep = new Set(want.map(t => t.join('_')));
    if (this.loaded.size > 320)
      for (const [k, g] of this.loaded) if (g && !keep.has(k)) { this.root.remove(g); g.traverse(o => o.geometry && o.geometry !== treeGeo && o.geometry.dispose()); this.loaded.delete(k); }
    for (const [i, j] of want) {
      if (this.busy >= 4) break;
      const k = `${i}_${j}`;
      if (this.loaded.has(k)) continue;
      this.loaded.set(k, null);
      this.busy++;
      fetch(`data/tiles/${k}.json`).then(r => r.json()).then(t => {
        if (!this.loaded.has(k)) return;
        const g = new THREE.Group(), trees = [];
        for (let q = 0; q < (t.t || []).length; q += 2) trees.push(t.t[q], t.t[q + 1], 0);
        for (const [q, x, y] of t.p || []) if (/conifer|broadleaf/.test(m.pkinds[q])) trees.push(x, y, m.pkinds[q] === 'conifer' ? 1 : 2);
        const fine = [], coarse = [];
        for (const [h, kind, f] of t.b || []) {
          const p = ring(f);
          if (p.length < 3) continue;
          const cx = p.reduce((a, v) => a + v.x, 0) / p.length, cy = p.reduce((a, v) => a + v.y, 0) / p.length;
          (this.hidden.some(q => inside(q, cx, cy)) ? coarse : fine).push([h, kind, p]);
        }
        g.add(ground(t.g || [], m.gkinds, s, i, j, trees), roads(t.r || [], m.rkinds, false), roads(t.r || [], m.rkinds, true), g.userData.b = buildings(fine, this.whole.has(k) ? coarseMat : buildingMat));
        if (coarse.length) { g.userData.lod = buildings(coarse, coarseMat); g.add(g.userData.lod); g.userData.lod.visible = this.lod > 0; }
        if (trees.length) g.add(forest(trees));
        g.name = k;
        g.userData.trees = trees;
        this.onTile?.(t, g);
        g.userData.res = (t.r || []).filter(([q]) => /residential|service|living|unclassified/.test(m.rkinds[q])).map(([q, , f]) => ({ s: m.rkinds[q] === 'service', g: f.reduce((a, v, i) => (i % 2 ? a[a.length - 1].push(v) : a.push([v]), a), []) }));
        this.root.add(g);
        this.loaded.set(k, g);
      }).catch(() => {}).finally(() => this.busy--);
    }
  }

  swap(k, on) {
    this.whole[on ? 'add' : 'delete'](k);
    const g = this.loaded.get(k);
    if (g) dress(g.userData.b, on ? coarseMat : buildingMat);
  }

  setLod(t) {
    this.lod = t;
    for (const g of this.loaded.values()) if (g?.userData.lod) g.userData.lod.visible = t > 0;
  }

  hide(polys) {
    this.hidden = this.hidden.concat(polys.map(q => q.map(([x, y]) => new THREE.Vector2(x, y))));
    const s = this.meta ? this.meta.size : 1000;
    for (const [k, g] of this.loaded)
      if (g && polys.some(q => q.some(([x, y]) => Math.floor(x / s) + '_' + Math.floor(y / s) === k))) { this.root.remove(g); this.loaded.delete(k); }
  }

  show(mode, zoom) {
    groundMat.visible = roadMat.visible = overMat.visible = mode !== 'flow';
    borderMat.uniforms.uA.value = mode === 'flow' ? 0.5 : 1;
    looks.px.value = 1 / zoom;
    const ramp = (a, b) => Math.min(Math.max((zoom - a) / (b - a), 0), 1);
    looks.tree.value = ramp(0.45, 0.75);
    looks.path.value = ramp(0.7, 1.1);
    looks.lane.value = ramp(2.5, 4);
    pathMat.visible = mode !== 'flow' && looks.path.value > 0;
    treeMat.visible = mode !== 'flow' && looks.tree.value > 0 && looks.far.value > 0 && looks.trees.value > 0;
  }
}
