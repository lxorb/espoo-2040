import * as THREE from 'three';

export const MATS = ['asphalt', 'asphalt_worn', 'paving', 'gravel', 'grass', 'meadow', 'forest_floor', 'scrub', 'sand', 'rock', 'parking', 'pitch', 'rail', 'water', 'sidewalk'];
export const L = Object.fromEntries(MATS.map((n, i) => [n, i]));
const MET = { asphalt: 6, asphalt_worn: 8, paving: 4, gravel: 6, grass: 12, meadow: 16, forest_floor: 10, scrub: 10, sand: 8, rock: 12, parking: 10, pitch: 20, rail: 2.4, water: 24, sidewalk: 8 };
const ONE = new Set(['paving', 'parking', 'sidewalk']);
const S = 512;

const hh = (x, y, s) => { let h = Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(s, 1274126177); h = Math.imul(h ^ h >>> 13, 1274126177); return ((h ^ h >>> 16) >>> 0) / 4294967296; };
const lerp = (a, b, t) => a + (b - a) * t;
const vn = (u, v, px, py, s) => {
  const x = u * px, y = v * py, i = Math.floor(x), j = Math.floor(y), a = i % px, b = j % py, a1 = (a + 1) % px, b1 = (b + 1) % py;
  let fx = x - i, fy = y - j; fx *= fx * (3 - 2 * fx); fy *= fy * (3 - 2 * fy);
  return lerp(lerp(hh(a, b, s), hh(a1, b, s), fx), lerp(hh(a, b1, s), hh(a1, b1, s), fx), fy);
};
const fbm = (u, v, p, n, s) => { let t = 0, a = 1, w = 0; for (let k = 0; k < n; k++, p *= 2, a *= 0.5) { t += a * vn(u, v, p, p, s + k); w += a; } return t / w; };
const mix = (a, b, t) => a.map((x, i) => lerp(x, b[i], t));
const ss = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const grey = g => [g, g * 1.01, g * 1.03];
const grass = (u, v, r, st) => mix([0.31, 0.43, 0.17], [0.47, 0.53, 0.26], fbm(u, v, 6, 4, 8)).map(c => c * st * (0.86 + 0.28 * vn(u, v, 256, 24, 9)) * (0.96 + 0.08 * r));

const GEN = {
  asphalt: (u, v, r) => grey(0.25 + 0.07 * fbm(u, v, 8, 3, 1) + 0.06 * (r - 0.5) + (r > 0.985 ? 0.14 : 0)),
  asphalt_worn: (u, v, r) => grey((0.38 + 0.14 * fbm(u, v, 4, 4, 2) + 0.06 * (r - 0.5)) * (Math.abs(fbm(u, v, 6, 3, 3) - 0.5) < 0.012 ? 0.55 : 1)),
  paving: (u, v, r) => {
    const y = v * 20, row = Math.floor(y), x = u * 10 + (row % 2) * 0.5, col = Math.floor(x), fx = x - col, fy = y - row;
    const t = (0.92 + 0.16 * hh(col % 10, row, 5) + 0.05 * (r - 0.5)) * (fx < 0.03 || fx > 0.97 || fy < 0.06 || fy > 0.94 ? 0.62 : 1);
    return [0.62 * t, 0.6 * t, 0.56 * t];
  },
  gravel: (u, v, r) => mix([0.5, 0.48, 0.44], [0.72, 0.69, 0.62], hh(Math.floor(u * 160), Math.floor(v * 160), 6)).map(c => c * (0.88 + 0.22 * fbm(u, v, 8, 3, 7)) * (0.94 + 0.12 * r)),
  grass: (u, v, r) => grass(u, v, r, 1 + 0.05 * Math.tanh(3 * Math.sin(v * Math.PI * 4))),
  meadow: (u, v, r) => mix([0.4, 0.47, 0.22], [0.62, 0.6, 0.37], fbm(u, v, 8, 4, 10)).map(c => c * (0.82 + 0.34 * vn(u, v, 192, 48, 11)) * (0.94 + 0.12 * r)),
  forest_floor: (u, v, r) => {
    const n = fbm(u, v, 6, 4, 12), m = fbm(u, v, 12, 3, 13);
    return mix(mix([0.31, 0.26, 0.17], [0.29, 0.37, 0.16], ss(0.4, 0.6, n)), [0.16, 0.14, 0.1], ss(0.58, 0.68, m)).map(c => c * (r > 0.97 ? 1.35 : 0.92 + 0.12 * r));
  },
  scrub: (u, v, r) => { const n = fbm(u, v, 16, 3, 14); return mix([0.17, 0.25, 0.11], [0.37, 0.47, 0.2], n).map(c => c * (n > 0.52 ? 1 : 0.7) * (0.92 + 0.16 * r)); },
  sand: (u, v, r) => [0.76, 0.7, 0.56].map(c => c * (0.9 + 0.14 * fbm(u, v, 8, 3, 15) + 0.08 * (r - 0.5))),
  rock: (u, v, r) => grey((0.48 + 0.24 * fbm(u, v, 4, 5, 16) + 0.05 * (r - 0.5)) * (Math.abs(fbm(u, v, 3, 4, 17) - 0.5) < 0.01 ? 0.6 : 1)),
  parking: (u, v, r) => (Math.abs(v - 0.5) < 0.006 || (v < 0.5 && (u * 4 % 1 < 0.012 || u * 4 % 1 > 0.988))) ? grey(0.85) : GEN.asphalt(u, v, r),
  pitch: (u, v, r) => grass(u, v, r, Math.floor(v * 4) % 2 ? 1.08 : 0.92),
  rail: (u, v, r) => {
    const d = Math.min(Math.abs(v - 0.317), Math.abs(v - 0.683));
    if (d < 0.015) return grey(d < 0.006 ? 0.72 : 0.38);
    if (u * 4 % 1 < 0.4) return [0.33, 0.28, 0.22].map(c => c * (0.9 + 0.2 * r));
    return GEN.gravel(u, v, r).map(c => c * 0.85);
  },
  water: (u, v) => grey(0.4 + 0.3 * vn(u, v, 6, 40, 18)),
  sidewalk: (u, v, r) => GEN.paving(u, v, r)
};

const lin = Array.from({ length: 256 }, (_, i) => (i /= 255) < 0.04045 ? i / 12.92 : ((i + 0.055) / 1.055) ** 2.4);

function gen(name, out) {
  const f = GEN[name];
  for (let y = 0, o = 0; y < S; y++) for (let x = 0; x < S; x++, o += 4) {
    const c = f(x / S, y / S, hh(x, y, 99));
    for (let k = 0; k < 3; k++) out[o + k] = Math.max(0, Math.min(255, Math.round(c[k] * 255)));
    out[o + 3] = 255;
  }
}

async function file(name) {
  const r = await fetch(`assets/ground/${name}.webp`);
  if (!r.ok) throw 0;
  const img = await createImageBitmap(await r.blob(), { resizeWidth: S, resizeHeight: S, resizeQuality: 'high' }), c = new OffscreenCanvas(S, S).getContext('2d');
  c.drawImage(img, 0, 0);
  img.close();
  return c.getImageData(0, 0, S, S).data;
}

export function materials(looks) {
  looks.mats.value = new THREE.DataArrayTexture(new Uint8Array(4), 1, 1, 1);
  (async () => {
    const list = await fetch('assets/ground/materials.json').then(r => r.json()).then(j => j.materials).catch(() => []), at = [];
    MATS.forEach((name, l) => {
      const m = list.find(q => q.name === name), n = Math.min(m?.variants || 1, 4);
      looks.var.value[l].set(at.length, n, +ONE.has(name), 0);
      for (let k = 0; k < n; k++) at.push([name, n > 1 ? `${name}-${k}` : name, l, m]);
    });
    const data = new Uint8Array(S * S * 4 * at.length), sum = MATS.map(() => [0, 0, 0, 0]);
    await Promise.all(at.map(async ([name, file_, l], k) => {
      const out = data.subarray(k * S * S * 4, (k + 1) * S * S * 4);
      try { out.set(await file(file_)); } catch { gen(name, out); }
      for (let o = 0; o < out.length; o += 4) for (let c = 0; c < 3; c++) sum[l][c] += lin[out[o + c]];
      sum[l][3] += S * S;
    }));
    MATS.forEach((name, l) => looks.mat.value[l].set(...sum[l].slice(0, 3).map(v => Math.max(v / sum[l][3], 1e-3)), list.find(q => q.name === name)?.metres || MET[name]));
    const tex = new THREE.DataArrayTexture(data, S, S, at.length);
    Object.assign(tex, { colorSpace: THREE.SRGBColorSpace, wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: true, anisotropy: 4, needsUpdate: true });
    looks.mats.value = tex;
    looks.ready.value = 1;
  })();
}

export const DETAIL = `uniform highp sampler2DArray uMats; uniform vec4 uMat[${MATS.length}], uVar[${MATS.length}]; uniform float uDetail;
  float gh(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  float gn(vec2 p) { vec2 i = floor(p), f = fract(p); f *= f * (3.0 - 2.0 * f); return mix(mix(gh(i), gh(i + vec2(1.0, 0.0)), f.x), mix(gh(i + vec2(0.0, 1.0)), gh(i + 1.0), f.x), f.y); }
  vec3 tap(vec2 u, vec2 dx, vec2 dy, vec4 v, float k) { return textureGrad(uMats, vec3(u, v.x + min(floor(k * v.y), v.y - 1.0)), dx, dy).rgb; }
  vec4 cell(vec2 u, vec2 dx, vec2 dy, vec4 v, float s, vec2 o) {
    vec2 q = u * s + o, c = floor(q), e = abs(fract(q) - 0.5);
    float h = gh(c + o * 17.0), r = floor(fract(h * 7.31) * 4.0);
    mat2 R = r < 1.0 ? mat2(1.0) : r < 2.0 ? mat2(0.0, 1.0, -1.0, 0.0) : r < 3.0 ? mat2(-1.0) : mat2(0.0, -1.0, 1.0, 0.0);
    return vec4(tap(R * u, R * dx, R * dy, v, h), smoothstep(0.0, 0.12, 0.5 - max(e.x, e.y)) + 1e-3);
  }
  vec3 raw(vec2 u, float l, float k) { return tap(u, dFdx(u), dFdy(u), uVar[int(l + 0.5)], k); }
  vec3 detail(vec2 p, vec2 w, float l, float k) {
    int i = int(l + 0.5); vec4 m = uMat[i], v = uVar[i];
    vec2 u = p / m.a, dx = dFdx(u), dy = dFdy(u);
    float f = smoothstep(0.08, 0.4, length(fwidth(u)));
    vec3 a = vec3(0.0);
    if (f < 1.0) {
      if (v.z > 0.5 || v.y < 1.5) a = tap(u, dx, dy, v, k < 0.0 ? 0.0 : k);
      else { vec4 A = cell(u, dx, dy, v, 1.0, vec2(0.0)), B = cell(u, dx, dy, v, 0.707, vec2(0.31, 0.57)); a = (A.rgb * A.a + B.rgb * B.a) / (A.a + B.a); }
    }
    if (f > 0.0) a = mix(a, tap(u / 6.0 + 0.3, dx / 6.0, dy / 6.0, v, 0.0), f);
    return a / m.rgb * (0.9 + 0.2 * gn(w / 47.0 + 5.0)) * (0.92 + 0.16 * gn(w / 211.0 + 3.0));
  }`;
