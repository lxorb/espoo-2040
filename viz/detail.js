import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { looks, patch, fadeDepth } from './city.js';
import { stamp } from './near.js';

const DETAIL_CAP = 12;
const v = new THREE.Vector3(), depth = fadeDepth(looks.fine);

export class Detail {
  constructor(scene, city) {
    Object.assign(this, { scene, city, tiles: new Map(), spare: new Map(), shared: new Map(), busy: 0, key: '', loader: new GLTFLoader() });
    this.use(false);
  }

  use(sat) {
    for (const k of [...this.tiles.keys()]) this.drop(k);
    for (const g of this.spare.values()) this.free(g);
    this.spare.clear();
    this.meta = null;
    this.key = '';
    const get = src => fetch(`data/${src}.json`).then(r => r.json()).then(m => [src, m]);
    const want = this.want = sat ? 'detail' : 'detail-clean';
    get(want).catch(() => get('detail')).then(([src, m]) => {
      if (want !== this.want) return;
      this.src = src;
      this.meta = m;
      this.part = new Set((m.partial || []).map(t => t.join('_')));
    }).catch(() => {});
  }

  update(camera, lod) {
    const m = this.meta;
    if (!m) return;
    for (const t of this.tiles.values()) if (t.g) t.g.visible = lod < 1;
    const s = m.size, on = lod < 1, key = stamp(camera, on, this.busy);
    if (key === this.key) return;
    this.key = key;
    const r = 2 * camera.zoom / innerWidth * s * 0.75, want = on ? m.tiles.map(([i, j]) => {
      v.set((i + 0.5) * s, 0, -(j + 0.5) * s).project(camera);
      return [v.x * v.x + v.y * v.y, `${i}_${j}`, Math.abs(v.x) < 1 + r && Math.abs(v.y) < 1 + r];
    }).filter(w => w[2]).sort((a, b) => a[0] - b[0]).slice(0, DETAIL_CAP) : [];
    const keep = new Set(want.map(w => w[1]));
    for (const k of [...this.tiles.keys()]) if (!keep.has(k)) this.drop(k);
    for (const [, k] of want) if (!this.tiles.has(k) && (this.spare.has(k) || this.busy < 2)) this.load(k);
  }

  load(k) {
    const g = this.spare.get(k);
    if (g) {
      this.spare.delete(k);
      this.tiles.set(k, { g });
      this.scene.add(g);
      if (!this.part.has(k)) this.city.swap(k, true);
      return;
    }
    const t = {};
    this.tiles.set(k, t);
    this.busy++;
    const src = this.src;
    this.loader.load(`data/${src}/${k}.glb`, g => {
      this.busy--;
      if (this.tiles.get(k) !== t || src !== this.src) return this.free(g.scene);
      const P = g.parser, J = P.json, keep = new Set(this.shared.values());
      g.scene.traverse(o => {
        if (!o.isMesh) return;
        for (const mt of [o.material].flat()) for (const [key, val] of Object.entries(mt)) {
          const a = val?.isTexture && P.associations.get(val), uri = a && J.images?.[J.textures?.[a.textures]?.source]?.uri;
          if (!uri) continue;
          const have = this.shared.get(uri);
          if (!have) this.shared.set(uri, val);
          else if (have !== val) { if (!keep.has(val)) val.dispose(); mt[key] = have; }
        }
        o.castShadow = o.receiveShadow = true;
        o.customDepthMaterial = depth;
        o.material = [o.material].flat().map(mt => { mt.flatShading = true; return patch(mt, '', 'diffuseColor.rgb *= uDim;', '', looks.one, looks.fine); });
        if (o.material.length === 1) o.material = o.material[0];
      });
      t.g = g.scene;
      this.scene.add(g.scene);
      if (!this.part.has(k)) this.city.swap(k, true);
      this.key = '';
    }, undefined, () => { this.busy--; this.key = ''; });
  }

  free(o) {
    o.traverse(m => {
      if (!m.isMesh) return;
      m.geometry.dispose();
      const keep = new Set(this.shared.values());
      for (const mt of [m.material].flat()) { for (const val of Object.values(mt)) if (val?.isTexture && !keep.has(val)) val.dispose(); mt.dispose(); }
    });
  }

  drop(k) {
    const t = this.tiles.get(k);
    this.tiles.delete(k);
    this.city.swap(k, false);
    if (!t.g) return;
    this.scene.remove(t.g);
    this.spare.set(k, t.g);
    for (const [q, g] of this.spare) if (this.spare.size > DETAIL_CAP * 2) { this.spare.delete(q); this.free(g); }
  }
}
