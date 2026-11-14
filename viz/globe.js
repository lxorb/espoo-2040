import * as THREE from 'three';

const LAT = 60.2 * Math.PI / 180, LON = 24.66 * Math.PI / 180, ease = t => t * t * t * (t * (6 * t - 15) + 10);
const VERT = `varying vec2 vU; varying vec3 vN, vV;
  void main() { vU = uv; vec4 p = modelViewMatrix * vec4(position, 1.0); vN = normalMatrix * normal; vV = -p.xyz; gl_Position = projectionMatrix * p; }`;

export function globe() {
  const cv = Object.assign(document.createElement('canvas'), { id: 'pg', className: 'pr' });
  const r = new THREE.WebGLRenderer({ canvas: cv, antialias: true }), scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(30, 1, 0.05, 50), tilt = new THREE.Group();
  const map = new THREE.TextureLoader().load('assets/globe.webp', t => r.initTexture(t));
  Object.assign(map, { colorSpace: THREE.SRGBColorSpace, anisotropy: 8 });
  const ball = new THREE.Mesh(new THREE.SphereGeometry(1, 160, 80), new THREE.ShaderMaterial({ uniforms: { map: { value: map } }, vertexShader: VERT,
    fragmentShader: `uniform sampler2D map; varying vec2 vU; varying vec3 vN, vV;
      void main() {
        float f = 1.0 - max(dot(normalize(vN), normalize(vV)), 0.0);
        gl_FragColor = vec4(texture2D(map, vU).rgb * (1.0 - 0.45 * f * f) + vec3(0.2, 0.34, 0.46) * pow(f, 3.5) * 0.5, 1.0);
        #include <colorspace_fragment>
      }` }));
  const air = new THREE.Mesh(new THREE.SphereGeometry(1.1, 96, 48), new THREE.ShaderMaterial({ vertexShader: VERT, side: THREE.BackSide, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    fragmentShader: `varying vec3 vN, vV;
      void main() { float g = clamp(-dot(normalize(vN), normalize(vV)) / 0.42, 0.0, 1.0); gl_FragColor = vec4(vec3(0.2, 0.34, 0.46) * g * g * 0.5, 1.0); }` }));
  scene.add(tilt.add(ball), air);
  document.body.append(cv);
  let on = false, t0 = 0, last = 0, spin = 0;
  const size = () => { r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.setSize(innerWidth, innerHeight, false); cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); };
  addEventListener('resize', size);
  function loop(now) {
    if (!on) return;
    const k = ease(Math.min((now - t0) / 6000, 1));
    spin += Math.min(now - last, 100) / 1000 * 0.05 * k;
    last = now;
    cam.position.z = 1 + 0.6 * (4.2 / 0.6) ** k;
    tilt.rotation.x = LAT + (0.42 - LAT) * k;
    ball.rotation.y = -Math.PI / 2 - LON + spin;
    r.render(scene, cam);
    requestAnimationFrame(loop);
  }
  return {
    show() { if (on) return; on = true; spin = 0; size(); t0 = last = performance.now(); cv.classList.add('on'); requestAnimationFrame(loop); },
    hide() { on = false; cv.classList.remove('on'); }
  };
}
