import * as THREE from 'three';

export const MATS = ['asphalt', 'asphalt_worn', 'paving', 'gravel', 'grass', 'meadow', 'forest_floor', 'scrub', 'sand', 'rock', 'parking', 'pitch', 'rail', 'water', 'sidewalk'];
export const L = Object.fromEntries(MATS.map((n, i) => [n, i]));
const ONE = new Set(['paving', 'parking', 'sidewalk']);
const S = 512;
const lin = Array.from({ length: 256 }, (_, i) => (i /= 255) < 0.04045 ? i / 12.92 : ((i + 0.055) / 1.055) ** 2.4);

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
      for (let k = 0; k < n; k++) at.push([n > 1 ? `${name}-${k}` : name, l]);
    });
    const data = new Uint8Array(S * S * 4 * at.length), sum = MATS.map(() => [0, 0, 0, 0]);
    await Promise.all(at.map(async ([name, l], k) => {
      const out = data.subarray(k * S * S * 4, (k + 1) * S * S * 4);
      try { out.set(await file(name)); } catch { out.fill(128); }
      for (let o = 0; o < out.length; o += 4) for (let c = 0; c < 3; c++) sum[l][c] += lin[out[o + c]];
      sum[l][3] += S * S;
    }));
    MATS.forEach((name, l) => looks.mat.value[l].set(...sum[l].slice(0, 3).map(v => Math.max(v / sum[l][3], 1e-3)), list.find(q => q.name === name)?.metres || 8));
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
