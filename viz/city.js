import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { MATS, L, DETAIL, materials } from './ground.js';

export const HYPE = [-1511.7, 1239.2];
export const looks = { dim: { value: 1 }, sat: { value: 1 }, tree: { value: 1 }, lane: { value: 1 }, path: { value: 1 }, one: { value: 1 }, px: { value: 1 }, coarse: { value: 0 }, fine: { value: 1 }, near: { value: 0 }, far: { value: 1 }, out: { value: 0.3 }, mark: { value: new THREE.Vector3(1, 1, 1) }, trees: { value: 1 }, sky: { value: new THREE.Vector3() },
  cover: { value: new THREE.DataTexture(new Uint8Array(4), 1, 1) }, cbox: { value: new THREE.Vector4() },
  mats: { value: null }, mat: { value: MATS.map(() => new THREE.Vector4(1, 1, 1, 8)) }, var: { value: MATS.map(() => new THREE.Vector4(0, 1, 0, 0)) }, detail: { value: 0 }, ready: { value: 0 }, gtex: { value: 0 } };
materials(looks);

const GROUND = { land: '#8e8f8b', water: '#15191c', forest: '#4f6339', park: '#667f48', grass: '#6f864e', field: '#959872',
  wetland: '#6d7a58', sand: '#b5ad92', rock: '#97958f', paved: '#9a9b97', rail: '#7d7a74', pitch: '#7a9c56',
  parking: '#8b8c88', playground: '#bfa98a', scrub: '#5f7342', garden: '#6b8449', hedge: '#3e5530', sidewalk: '#a9aaa6', conifer_forest: '#3d5236' };
const NEAR = { ...GROUND, forest: '#505834', conifer_forest: '#464e30', park: '#646c3c', grass: '#666e3c', field: '#7a7650', wetland: '#646a4a', pitch: '#6c7541',
  scrub: '#575e37', garden: '#656d3c', hedge: '#3e4628' };
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
    Object.assign(s.uniforms, { uDim: looks.dim, uSat: looks.sat, uTree: looks.tree, uLane: looks.lane, uFade: fade, uPx: looks.px, uShow: show, uSky: looks.sky, uCover: looks.cover, uCbox: looks.cbox, uMats: looks.mats, uMat: looks.mat, uVar: looks.var, uDetail: looks.detail });
    s.vertexShader = s.vertexShader.replace('#include <common>', `#include <common>\nuniform float uTree;\n${pars}`).replace('#include <begin_vertex>', `#include <begin_vertex>\n${vert}`);
    s.fragmentShader = s.fragmentShader.replace('#include <common>', `#include <common>\nuniform float uDim, uSat, uLane, uFade, uShow;\nuniform vec3 uSky;\n${hash}\n${COVER}\n${DETAIL}\n${pars.replace(/attribute[^;]*;/g, '')}`)
      .replace('#include <dithering_fragment>', '#include <dithering_fragment>\nif (sky > 0.5) gl_FragColor.rgb = uSky * sky;')
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

const RIB = ['attribute vec3 aB, aR, aJ; varying vec3 vB, vR, vJ; varying vec2 vW; float line(float d, float h, float f) { float e = max(h, 0.6 * f); return (1.0 - smoothstep(e - 0.5 * f, e + 0.5 * f, d)) * h / e; }',
  'vB = aB; vR = aR; vJ = aJ; vW = (modelMatrix * vec4(position, 1.0)).xz;', `if (${DITHER} < cover(vW)) discard;
  float w = vR.z, x = vB.y * 0.5 * w, n = vB.z, fw = fwidth(x), fu = max(fwidth(vB.x), 1e-4);`];
const roadLook = fade => patch(new THREE.MeshStandardMaterial({ roughness: 0.9, vertexColors: true }), RIB[1], `${RIB[2]}
  if (uDetail > 0.0) diffuseColor.rgb = mix(diffuseColor.rgb, (vR.x < 0.5 ? vec3(0.091, 0.093, 0.098) : diffuseColor.rgb) * detail(vec2(vB.x, x), vec2(vW.x, -vW.y), vR.x, -1.0), uDetail);
  float t = (x + 0.5 * w - 0.5) / 3.3, q = mod(vB.x, 12.0), major = step(2.0, mod(vR.y, 4.0));
  float dash = mix(clamp(min(q, 3.0 - q) / fu + 0.5, 0.0, 1.0), 0.25, smoothstep(2.0, 6.0, fu));
  float m = major * max(step(1.5, n) * line(abs(t - clamp(floor(t + 0.5), 1.0, n - 1.0)) * 3.3, 0.07, fw) * dash, line(abs(abs(x) - 0.5 * w + 0.5), 0.08, fw));
  m *= smoothstep(vJ.x - 2.0, vJ.x, vB.x) * smoothstep(vJ.y - 2.0, vJ.y, vJ.z - vB.x);
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.9, 0.9, 0.87), uLane * m);
  diffuseColor.rgb = mix(vec3(0.27, 0.275, 0.26), diffuseColor.rgb, uFade) * uDim;`, RIB[0], fade);
const roadMat = roadLook(looks.one), pathMat = roadLook(looks.path);
const sideMat = patch(new THREE.MeshStandardMaterial({ roughness: 1, vertexColors: true }), RIB[1], `${RIB[2]}
  float r = 0.5 * w - 2.5, ax = abs(x);
  if (ax < r - 0.2) discard;
  diffuseColor.rgb = raw(vec2((ax - r) / 2.5, vB.x / uMat[${L.sidewalk}].a), ${L.sidewalk}.0, vR.y) * uDim;`, RIB[0], looks.one, looks.detail);
export const borderMat = new THREE.ShaderMaterial({ uniforms: { uPx: looks.px, uA: { value: 1 } },
  vertexShader: 'attribute vec3 aE; uniform float uPx; void main() { vec3 p = position; p.xz += aE.xy * aE.z * uPx * 0.5; gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); }',
  fragmentShader: 'uniform float uA; void main() { gl_FragColor = vec4(vec3(0.254, 0.254, 0.235) * uA, 1.0); }' });
const overMat = patch(new THREE.MeshStandardMaterial({ roughness: 0.9, vertexColors: true, depthTest: false, depthWrite: false }),
  'transformed.xz += aE.xy * max(0.0, uPx * 2.5 - aE.z) * 0.5;', 'diffuseColor.rgb *= uDim;', 'attribute vec3 aE; uniform float uPx;');

const groundMat = patch(new THREE.MeshStandardMaterial({ roughness: 1, vertexColors: true, depthWrite: false, depthTest: false }), 'vG = aG; vC = aC; vW = (modelMatrix * vec4(position, 1.0)).xz;', `
  diffuseColor.rgb *= uDim; sky = step(distance(vColor, vec3(${BASE.toArray().map(v => v.toFixed(5))})), 0.001);
  if (uDetail > 0.0) {
    vec2 p = vec2(vW.x, -vW.y); float c = cos(vG.y), s = sin(vG.y);
    vec2 q = mat2(c, -s, s, c) * p; vec3 d = detail(q, p, vG.x, vG.w);
    if (vG.z > 0.5) d = mix(d, detail(q, p, vG.z > 1.5 ? ${L.grass}.0 : ${L.meadow}.0, -1.0), smoothstep(0.3, 0.7, gn(p / 90.0 + 11.0)));
    vec3 n = vG.z > 0.5 && vG.z < 1.5 ? mix(vec3(0.134, 0.157, 0.048), vec3(0.171, 0.155, 0.061), 0.8 * gn(p / 80.0 + 7.0)) * (0.85 + 0.25 * gn(p / 31.0)) : vC;
    if (sky > 0.5) sky = 1.0 + 0.25 * uDetail * (d.g - 1.0);
    else diffuseColor.rgb = mix(diffuseColor.rgb, n * uDim * d, uDetail);
  }`, 'attribute vec4 aG; attribute vec3 aC; varying vec4 vG; varying vec3 vC; varying vec2 vW;');
const GL = { land: 'grass', water: 'water', forest: 'forest_floor', park: 'grass', grass: 'grass', field: 'meadow', wetland: 'meadow', sand: 'sand', rock: 'rock', paved: 'paving', rail: 'rail',
  pitch: 'pitch', parking: 'parking', playground: 'sand', garden: 'grass', scrub: 'scrub', hedge: 'scrub', sidewalk: 'paving', conifer_forest: 'forest_floor' };
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
  const pos = [], col = [], ag = [], an = [], c = new THREE.Color(), nc = new THREE.Color();
  const quad = (x0, y0, x1, y1, h, cc) => { for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]) { pos.push(x, h, -y); col.push(cc.r, cc.g, cc.b); an.push(cc.r, cc.g, cc.b); ag.push(L.water, 0, 0, 0); } };
  quad(i * size, j * size, (i + 1) * size, (j + 1) * size, -0.05, BASE);
  const polys = list.map(([k, f]) => {
    const p = ring(f), b = [Infinity, Infinity, -Infinity, -Infinity];
    for (const v of p) { b[0] = Math.min(b[0], v.x); b[1] = Math.min(b[1], v.y); b[2] = Math.max(b[2], v.x); b[3] = Math.max(b[3], v.y); }
    return { name: kinds[k] || '', p, b };
  }).filter(q => q.p.length > 2);
  const covered = (from, x, y) => polys.some((q, k) => k > from && !SCATTER[q.name] && x > q.b[0] && x < q.b[2] && y > q.b[1] && y < q.b[3] && inside(q.p, x, y));
  polys.forEach(({ name, p, b }, k) => {
    c.set(GROUND[name] || GROUND.land); nc.set(NEAR[name] || NEAR.land);
    let a = 0, best = 0, h = Math.abs(Math.sin(p[0].x * 12.9898 + p[0].y * 78.233) * 43758.5453) % 1;
    p.forEach((v, i) => { const u = p[(i + 1) % p.length], l = v.distanceToSquared(u); if (l > best) { best = l; a = Math.atan2(u.y - v.y, u.x - v.x); } });
    for (const t of tris(p))
      for (const v of t) { pos.push(p[v].x, 0, -p[v].y); col.push(c.r, c.g, c.b); an.push(nc.r, nc.g, nc.b); ag.push(L[GL[name] || 'grass'], a, name === 'land' || !name ? 1 : /forest/.test(name) ? 2 : 0, h); }
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
  g.setAttribute('aG', new THREE.Float32BufferAttribute(ag, 4));
  g.setAttribute('aC', new THREE.Float32BufferAttribute(an, 3));
  const m = new THREE.Mesh(up(g), groundMat);
  m.receiveShadow = true;
  m.renderOrder = -1;
  return m;
}

const roadStyle = k => /rail|tram|subway/.test(k) ? [2.4, '#55524d', 0.48] : /foot|cycle|track/.test(k) ? [1.2, '#979893', 0.4] : k === 'service' ? [3.5, '#7a7c7d', 0.42] : [0, '#5e6164', 0.45];

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

const RL = k => L[/rail|tram|subway/.test(k) ? 'rail' : /track/.test(k) ? 'gravel' : /foot/.test(k) ? 'paving' : 'asphalt'];
const CAR = /motorway|trunk|primary|secondary|tertiary|residential/;
export const jkey = (x, y) => Math.round(x * 2) + ',' + Math.round(y * 2);
export function junctions(list, kinds) {
  const J = new Map();
  for (const [k, lanes, f] of list) if (CAR.test(kinds[k])) for (let i = 0, n = f.length / 2; i < n; i++) {
    const q = jkey(f[2 * i], f[2 * i + 1]), j = J.get(q) || J.set(q, { x: f[2 * i], y: f[2 * i + 1], d: 0, w: 0 }).get(q);
    j.d += i && i < n - 1 ? 2 : 1;
    j.w = Math.max(j.w, Math.max(lanes, 1) * 3.3 + 1);
  }
  for (const [q, j] of J) if (j.d < 3) J.delete(q);
  return J;
}
function rib(lines, mat) {
  const r = [], j = [], m = new THREE.Mesh(up(ribbon(lines, ({ r: q, j: e }) => { r.push(...q); j.push(...e); })), mat);
  m.geometry.setAttribute('aR', new THREE.Float32BufferAttribute(r, 3));
  m.geometry.setAttribute('aJ', new THREE.Float32BufferAttribute(j, 3));
  m.receiveShadow = true;
  return m;
}
function roads(list, kinds, minor) {
  const J = junctions(list, kinds), at = (x, y) => (J.get(jkey(x, y))?.w ?? -4) + 4;
  const lines = list.filter(([k]) => kinds[k] !== 'tunnel' && minor === /foot|cycle|service|track/.test(kinds[k])).flatMap(([k, lanes, f, fl = 0]) => {
    const name = kinds[k] || '', [w, color, y] = roadStyle(name), ww = w || Math.max(lanes, 1) * 3.3 + 1, n = f.length / 2, cut = [0];
    const base = { w: ww, color, y, lanes: w ? 0 : lanes, r: [RL(name), (fl & 1) + 2 * /motorway|trunk|primary|secondary|tertiary/.test(name), ww], walk: /primary|secondary|tertiary|residential/.test(name) && fl & 12 };
    if (CAR.test(name)) for (let i = 1; i < n - 1; i++) if (J.has(jkey(f[2 * i], f[2 * i + 1]))) cut.push(i);
    cut.push(n - 1);
    return cut.slice(1).map((b, c) => {
      const pts = f.slice(2 * cut[c], 2 * b + 2);
      let len = 0;
      for (let i = 2; i < pts.length; i += 2) len += Math.hypot(pts[i] - pts[i - 2], pts[i + 1] - pts[i - 1]);
      return { ...base, pts, j: [at(pts[0], pts[1]), at(pts[pts.length - 2], pts[pts.length - 1]), len] };
    });
  });
  const side = lines.filter(l => l.walk).map(l => ({ ...l, w: l.w + 5, y: 0.3, color: GROUND.sidewalk, r: [L.sidewalk, Math.random(), l.w + 5] }));
  return [rib(lines, minor ? pathMat : roadMat), ...side.length ? [rib(side, sideMat)] : []];
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
      const rings = (o.border || []).map(round);
      rings.forEach(f => {
        const q = [], m = new THREE.Mesh(ribbon([{ pts: f, w: 0, y: 0.6, px: 2 }], ({ px }, s, dx, dz) => q.push(-dz * s, -dx * s, px)), borderMat);
        m.geometry.setAttribute('aE', new THREE.Float32BufferAttribute(q, 3));
        m.frustumCulled = false;
        scene.add(m);
      });
      if (rings.length) {
        const ys = rings.flat().filter((v, i) => i % 2);
        const far = 4e5, shape = new THREE.Shape([[-far, -far], [far, -far], [far, far], [-far, far]].map(p => new THREE.Vector2(...p)));
        for (const f of rings) { const p = []; for (let i = 0; i < f.length - 2; i += 2) p.push(new THREE.Vector2(f[i], f[i + 1])); shape.holes.push(new THREE.Path(p)); }
        const shade = new THREE.Mesh(new THREE.ShapeGeometry(shape).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({ uniforms: { uOut: looks.out }, depthTest: false, depthWrite: false, transparent: true,
          blending: THREE.CustomBlending, blendSrc: THREE.ZeroFactor, blendDst: THREE.SrcColorFactor,
          vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }', fragmentShader: 'uniform float uOut; void main() { gl_FragColor = vec4(vec3(pow(uOut, 0.4545)), 1.0); }' }));
        shade.renderOrder = 50; shade.frustumCulled = false;
        scene.add(shade);
        const f = rings.reduce((a, b) => b.length > a.length ? b : a);
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
        g.add(ground(t.g || [], m.gkinds, s, i, j, trees), ...roads(t.r || [], m.rkinds, false), ...roads(t.r || [], m.rkinds, true), g.userData.b = buildings(fine, this.whole.has(k) ? coarseMat : buildingMat));
        if (coarse.length) { g.userData.lod = buildings(coarse, coarseMat); g.add(g.userData.lod); g.userData.lod.visible = this.lod > 0; }
        if (trees.length) g.add(forest(trees));
        g.name = k;
        g.userData.trees = trees;
        this.onTile?.(t, g);
        g.userData.res = (t.r || []).filter(([q]) => /residential|service/.test(m.rkinds[q])).map(([q, , f]) => ({ s: m.rkinds[q] === 'service', g: f.reduce((a, v, i) => (i % 2 ? a[a.length - 1].push(v) : a.push([v]), a), []) }));
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
    groundMat.visible = roadMat.visible = overMat.visible = sideMat.visible = mode !== 'flow';
    borderMat.uniforms.uA.value = mode === 'flow' ? 0.5 : 1;
    looks.px.value = 1 / zoom;
    const ramp = (a, b) => Math.min(Math.max((zoom - a) / (b - a), 0), 1);
    looks.tree.value = ramp(0.45, 0.75);
    looks.path.value = ramp(0.7, 1.1);
    looks.lane.value = ramp(0.8, 1.4);
    pathMat.visible = mode !== 'flow' && looks.path.value > 0;
    treeMat.visible = mode !== 'flow' && looks.tree.value > 0 && looks.far.value > 0 && looks.trees.value > 0;
  }
}
