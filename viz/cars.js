import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { lengths, along } from './traffic.js';

const SHAPES = {
  car: [4.5, 1.8, 0.325, [0.9, 3.55], [[[0, 0.3], [4.5, 0.3], [4.5, 0.66], [4.3, 0.8], [3.3, 0.92], [0.1, 0.98], [0, 0.9]], [[0.4, 0.95], [3.4, 0.9], [2.6, 1.36], [1.0, 1.4], [0.3, 1.0]], [[1.0, 1.36], [2.62, 1.33], [2.55, 1.44], [1.08, 1.46]]]],
  van: [5.2, 2, 0.36, [0.9, 4.2], [[[0, 0.3], [5.2, 0.3], [5.2, 1], [4.4, 1.15], [0, 1.15]], [[0.2, 1.1], [4.45, 1.1], [3.75, 1.9], [0.2, 1.9]], [[0, 1.12], [0.25, 1.12], [0.25, 1.88], [3.8, 1.88], [3.65, 2.15], [0, 2.15]]]],
  truck: [8.5, 2.5, 0.5, [1.3, 2.5, 7.3], [[[0, 0.6], [8.5, 0.6], [8.5, 1.5], [6.4, 1.5], [6.4, 3.4], [0, 3.4]], [[6.5, 1.45], [8.53, 1.45], [8.35, 2.4], [6.5, 2.4]], [[6.4, 2.35], [8.36, 2.35], [8.25, 2.8], [6.4, 2.8]]]],
  bus: [12, 2.55, 0.5, [2.5, 9.6], [[[0, 0.35], [12, 0.35], [12, 1.2], [0, 1.2]], [[0.3, 1.15], [12.03, 1.15], [11.95, 2.6], [0.3, 2.6]], [[0, 1.15], [0.3, 1.15], [0.3, 2.55], [12, 2.55], [11.85, 3.1], [0, 3.1]]]],
  moto: [2.1, 0.5, 0.32, [0.35, 1.75], [[[0.1, 0.4], [2, 0.4], [1.85, 0.8], [0.25, 0.85]], [[0.65, 0.8], [1.25, 0.8], [1.1, 1.5], [0.75, 1.5]]]]
};
const PAINT = { car: ['#e8e8e6', '#1c1d20', '#a9adb3', '#b3242a', '#2c4f8a', '#5d6267', '#d8d2c2', '#3d6b4f'], van: ['#f2f2ef', '#c9ccd0', '#2c4f8a'], truck: ['#f2f2ef', '#e07a1f', '#2c4f8a'], bus: ['#2a62b8'], moto: ['#1c1d20', '#b3242a'] };
const MAX = 20000, MIX = [0.82, 0.9, 0.96, 0.97, 1], hash = n => { const v = Math.sin(n * 12.9898) * 43758.5453; return v - Math.floor(v); };

function vehicle(t) {
  const [L, W, R, wheels, parts] = SHAPES[t] || SHAPES.car, out = [], tint = new THREE.Color();
  const add = (g, c, paint) => {
    g = g.index ? g.toNonIndexed() : g;
    g.deleteAttribute('uv');
    tint.set(c);
    g.setAttribute('color', new THREE.Float32BufferAttribute(Array(g.attributes.position.count).fill([tint.r, tint.g, tint.b]).flat(), 3));
    g.setAttribute('aPaint', new THREE.Float32BufferAttribute(Array(g.attributes.position.count).fill(paint), 1));
    out.push(g);
  };
  parts.forEach((pts, k) => {
    const w = t === 'moto' ? (k ? 0.45 : 0.3) : k === 1 ? W - 0.16 : k ? W - 0.1 : W;
    add(new THREE.ExtrudeGeometry(new THREE.Shape(pts.map(([x, y]) => new THREE.Vector2(x, y))), { depth: w, bevelEnabled: false }).translate(0, 0, -w / 2), k === 1 ? '#1d2226' : '#fff', k === 1 ? 0 : 1);
  });
  for (const x of wheels) for (const z of t === 'moto' ? [0] : [-1, 1])
    add(new THREE.CylinderGeometry(R, R, t === 'moto' ? 0.12 : 0.22, 12).rotateX(Math.PI / 2).translate(x, R, z * (W / 2 - 0.17)), '#2a2c2f', 0);
  for (const z of t === 'moto' ? [0] : [-1, 1]) {
    add(new THREE.BoxGeometry(0.06, 0.12, 0.3).translate(L, R + 0.3, z * (W / 2 - 0.3)), '#fff4d0', 0);
    add(new THREE.BoxGeometry(0.06, 0.12, 0.3).translate(0, R + 0.35, z * (W / 2 - 0.3)), '#a3141a', 0);
  }
  const box = new THREE.BoxGeometry(L, R + (t === 'moto' || t === 'car' ? 1.0 : 1.6), W);
  box.computeBoundingBox();
  add(box.translate(L / 2, -box.boundingBox.min.y, 0), '#fff', 1);
  const lo = out.pop();
  return [mergeGeometries(out).translate(-L / 2, 0, 0), lo.translate(-L / 2, 0, 0), L];
}
const carMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 });
carMat.onBeforeCompile = s => {
  s.vertexShader = s.vertexShader.replace('#include <common>', '#include <common>\nattribute float aPaint;').replace('#include <color_vertex>', 'vColor = mix(color.rgb, color.rgb * instanceColor, aPaint);');
};

export class Cars {
  constructor(scene, meta, pts) {
    this.meta = meta;
    this.max = MAX;
    this.pts = pts;
    this.end = Math.max(...meta.veh.map(v => v[1] + v[3]));
    this.t = 0;
    this.heading = new Float32Array(meta.veh.length);
    this.box = meta.box || [0, 1, 0, 1].map((a, k) => pts.filter((v, i) => i % 2 === a).reduce((m, v) => (k < 2 ? Math.min : Math.max)(m, v / 10), k < 2 ? Infinity : -Infinity));
    this.meshes = meta.types.map((t, k) => {
      const n = meta.veh.filter(v => v[0] === k).length;
      const [hi, lo, len] = vehicle(t), m = new THREE.InstancedMesh(hi, carMat, n + MAX);
      Object.assign(m.userData, { hi, lo, len });
      m.setColorAt(0, new THREE.Color());
      m.castShadow = true;
      m.frustumCulled = false;
      m.userData.paint = (PAINT[t] || PAINT.car).map(c => new THREE.Color(c));
      scene.add(m);
      return m;
    });
  }

  tick(dt, nets, rect, px, tiles) {
    this.t = (this.t + dt * this.meta.dt) % this.end;
    const o = new THREE.Object3D(), P = this.pts, count = this.meshes.map(() => 0);
    for (const m of this.meshes) { const s = m.userData.len / px; m.userData.off = s < 2; m.geometry = s < 10 ? m.userData.lo : m.userData.hi; }
    const put = (type, x, y, h, id) => {
      const m = this.meshes[type], c = count[type];
      if (c >= m.instanceMatrix.count || m.userData.off) return;
      count[type]++;
      o.position.set(x, 0.45, -y);
      o.rotation.y = h;
      o.updateMatrix();
      m.setMatrixAt(c, o.matrix);
      m.setColorAt(c, m.userData.paint[(id * 7919) % m.userData.paint.length]);
    };
    this.meta.veh.forEach(([type, start, off, n], id) => {
      const s = (this.t - start) / this.meta.dt, k = Math.floor(s);
      if (k < 0 || k >= n - 1) return;
      const a = 2 * (off + k), f = s - k;
      const dx = P[a + 2] - P[a], dy = P[a + 3] - P[a + 1], d = dx * dx + dy * dy;
      if (d > 2e5 * this.meta.dt ** 2) return;
      const x = (P[a] + dx * f) / 10, y = (P[a + 1] + dy * f) / 10;
      if (d > 4) this.heading[id] = Math.atan2(dy, dx);
      if (!nets.find(n => n.keep(x, y))?.blocked(x, y)) put(type, x, y, this.heading[id], id);
    });
    for (const n of nets) if (px < 2.25) this.macro(n, rect, dt, put); else if (n.carState) n.carState.mac = [];
    if (px < 2.25) this.streets(tiles, rect, put);
    this.meshes.forEach((m, k) => {
      m.count = count[k];
      m.instanceMatrix.needsUpdate = true;
      if (m.instanceColor) m.instanceColor.needsUpdate = true;
    });
  }

  show(mode) {
    for (const m of this.meshes) m.visible = mode === 'cars';
  }

  inBox(x, y) {
    const B = this.box;
    return x > B[0] && x < B[2] && y > B[1] && y < B[3];
  }

  macro(net, [x0, y0, x1, y1], dt, put) {
    const S = net.carState ||= { mac: [] }, L = net.links, n = L.length, e = [0, 0, 0, 0];
    if (L !== S.L) {
      const twin = new Set(L.map(l => l.a + ',' + l.b));
      S.L = L; S.mac = S.mac.filter(c => c.i < n); S.out = [];
      L.forEach((l, i) => (S.out[l.a] ||= []).push(i));
      S.cum = L.map(l => lengths(l.geom));
      S.two = L.map(l => twin.has(l.b + ',' + l.a));
      S.inside = L.map(l => this.inBox(...l.geom[l.geom.length >> 1]));
    }
    const vis = new Uint8Array(n), want = new Float32Array(n), on = new Map(), len = i => S.cum[i][S.cum[i].length - 1];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const b = net.box[i];
      if (S.inside[i] || !net.keep(...L[i].geom[L[i].geom.length >> 1]) || b[0] > x1 + 100 || b[2] < x0 - 100 || b[1] > y1 + 100 || b[3] < y0 - 100 || !(net.flow[i] > 0)) continue;
      vis[i] = 1;
      sum += want[i] = Math.min(net.flow[i] * len(i) / Math.max(net.speed[i], 0.5) / 3600, len(i) * Math.max(L[i].lanes, 1) / 8);
    }
    const k = Math.min(1, this.max / Math.max(sum, 1));
    const next = i => {
      const o = (S.out[L[i].b] || []).filter(j => vis[j]), tot = o.reduce((a, j) => a + net.flow[j], 0);
      let r = Math.random() * tot;
      return o.find(j => (r -= net.flow[j]) < 0) ?? -1;
    };
    S.mac = S.mac.filter(c => {
      if (!vis[c.i]) return false;
      c.s += net.speed[c.i] * dt;
      while (c.s > len(c.i)) { c.s -= len(c.i); c.i = next(c.i); if (c.i < 0) return false; }
      const l = on.get(c.i) || [];
      on.set(c.i, l);
      if (l.length > Math.floor(want[c.i] * k + hash(c.i)) + 1) return false;
      l.push(c);
      return true;
    });
    let budget = 400;
    for (let i = 0; i < n && budget > 0; i++) {
      if (!vis[i]) continue;
      const l = on.get(i) || [];
      if (l.length >= Math.floor(want[i] * k + hash(i))) continue;
      const c = { i, s: Math.random() * len(i), r: Math.random(), id: Math.random() * 1e6 | 0 };
      const lane = c.r * L[i].lanes | 0;
      if (l.some(d => (d.r * L[i].lanes | 0) === lane && Math.abs(d.s - c.s) < 12)) continue;
      c.type = MIX.findIndex(m => hash(c.id) < m);
      S.mac.push(c); budget--;
    }
    for (const c of S.mac) {
      const l = L[c.i], m = Math.max(l.lanes, 1), j = Math.min(c.r * m | 0, m - 1), off = (j + 0.5 - (S.two[c.i] ? 0 : m / 2)) * 3.3;
      along(l.geom, S.cum[c.i], c.s, e);
      if (net.keep(e[0], e[1])) put(c.type, e[0] + e[3] * off, e[1] - e[2] * off, Math.atan2(e[3], e[2]), c.id);
    }
  }

  streets(tiles, [x0, y0, x1, y1], put) {
    const e = [0, 0, 0, 0];
    for (const g of tiles.values()) {
      if (!g?.userData.res) continue;
      const [ti, tj] = g.name.split('_').map(Number);
      if (ti * 1000 > x1 || ti * 1000 + 1000 < x0 || tj * 1000 > y1 || tj * 1000 + 1000 < y0) continue;
      g.userData.res.forEach((f, q) => {
        const h = hash(ti * 7177 + tj * 131 + q);
        f.cum ||= lengths(f.g);
        const L = f.cum[f.cum.length - 1];
        if (h > 0.3 || L < 40) return;
        const u = (this.t * 5 + h * 1e4) % (2 * L), back = u > L;
        along(f.g, f.cum, back ? 2 * L - u : u, e);
        const dx = back ? -e[2] : e[2], dy = back ? -e[3] : e[3];
        if (this.inBox(e[0], e[1]) || e[0] < x0 || e[0] > x1 || e[1] < y0 || e[1] > y1) return;
        put(h < 0.02 ? 1 : 0, e[0] + dy * 1.4, e[1] - dx * 1.4, Math.atan2(dy, dx), q);
      });
    }
  }
}
