(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = { preview: null, yaw: -0.65, pitch: 0.45, zoom: 0.78, drag: null };
  const canvas = $('viewer');
  const ctx = canvas.getContext('2d');

  function value(id) { return $(id).value; }
  function number(id) { return Number(value(id)); }
  function node(id, block, params) { return { id, block, params }; }

  function buildGraph() {
    const nodes = [
      node('domain', 'cube_domain', { length: number('length'), mesh_density: number('density'), periodic: $('periodic').checked }),
      node('shapes', value('shape'), { n_particles: number('particles'), radius_mean: number('radius'), radius_std: number('radius-std') }),
      node('packing', 'rsa_packing', { packer: value('packer'), box_ratio: 1.35, gap_cells: 2, wall_cells: 1, seed: number('seed') }),
      node('template', 'macroblock_template', { element_type: value('element'), n_inner: 2, n_outer: 2 }),
      node('quality', 'quality_gate', { min_scaled_jacobian: number('jacobian'), max_edge_ratio: 12, max_volume_error: 0.05 }),
      node('output', 'exodus_output', { path: 'output/mesh.e', write_report: true, write_vtk: true }),
    ];
    const chain = ['template'];
    if ($('pillow').checked) { nodes.push(node('pillow', 'pillow_layer', { thickness_fraction: 0.15 })); chain.push('pillow'); }
    if ($('smoothing').checked) { nodes.push(node('smoothing', 'quality_smoothing', { n_iterations: 20, step_size: 0.5, min_scaled_jacobian: 0.5774 })); chain.push('smoothing'); }
    chain.push('quality', 'output');
    const edges = [{ from: 'domain', to: 'packing' }, { from: 'shapes', to: 'packing' }, { from: 'packing', to: 'template' }];
    for (let i = 0; i < chain.length - 1; i += 1) edges.push({ from: chain[i], to: chain[i + 1] });
    return { nodes, edges };
  }

  async function generate() {
    const button = $('generate');
    button.disabled = true;
    button.querySelector('span').textContent = 'Generating…';
    $('form-note').textContent = 'Packing, assembling, validating, and writing artifacts…';
    try {
      const response = await fetch('/api/graph/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(buildGraph()) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
      state.preview = data.preview;
      $('empty-state').hidden = true;
      render();
      showResult(data);
    } catch (error) {
      $('results').hidden = false;
      $('result-title').textContent = 'Generation failed';
      $('result-message').textContent = error.message;
      $('downloads').replaceChildren();
      $('report-json').textContent = JSON.stringify({ error: error.message }, null, 2);
      $('form-note').textContent = 'Correct the configuration and try again.';
    } finally {
      button.disabled = false;
      button.querySelector('span').textContent = 'Generate mesh';
    }
  }

  function showResult(data) {
    const report = data.report;
    const validation = report.validation || {};
    const quality = validation.quality || {};
    const passed = validation.passed !== false;
    const statValues = [report.mesh.n_elements.toLocaleString(), report.mesh.n_nodes.toLocaleString(), Number(quality.min_scaled_jacobian).toFixed(4), passed ? 'VALID' : 'INVALID'];
    [...$('stats').querySelectorAll('b')].forEach((el, i) => { el.textContent = statValues[i]; el.className = i === 3 ? (passed ? 'valid' : 'invalid') : ''; });
    $('results').hidden = false;
    $('result-title').textContent = passed ? 'Validated mesh ready' : 'Mesh generated with validation failures';
    $('result-message').textContent = passed ? `Seed ${report.seed} produced a conforming ${report.mesh.element_type} mesh in ${Number(report.timings.total).toFixed(2)} seconds.` : (validation.failures || []).join(' · ');
    const downloads = $('downloads'); downloads.replaceChildren();
    Object.entries(data.downloads).forEach(([name, href]) => { const a = document.createElement('a'); a.href = href; a.textContent = `Download ${name}`; downloads.appendChild(a); });
    $('report-json').textContent = JSON.stringify(report, null, 2);
    $('form-note').textContent = `Artifacts saved under output/gui/${data.token}.`;
    $('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function resize() {
    const rect = canvas.getBoundingClientRect(); const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; render(); }
  }

  function transform(point, center, scale) {
    let x = (point[0] - center[0]) * scale, y = (point[1] - center[1]) * scale, z = (point[2] - center[2]) * scale;
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw); [x, z] = [cy * x + sy * z, -sy * x + cy * z];
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch); [y, z] = [cp * y - sp * z, sp * y + cp * z];
    return [x, y, z];
  }

  function qualityColor(q, alpha) {
    const t = Math.max(0, Math.min(1, (q + 0.05) / 0.7));
    const low = [228, 91, 53], high = [0, 108, 100];
    return `rgba(${Math.round(low[0] + (high[0] - low[0]) * t)},${Math.round(low[1] + (high[1] - low[1]) * t)},${Math.round(low[2] + (high[2] - low[2]) * t)},${alpha})`;
  }

  function render() {
    const w = canvas.width, h = canvas.height; ctx.clearRect(0, 0, w, h);
    const p = state.preview; if (!p || !p.points.length) return;
    const center = p.bounds.length ? p.bounds[0].map((v, i) => (v + p.bounds[1][i]) / 2) : [0.5, 0.5, 0.5];
    const extent = p.bounds.length ? Math.max(...p.bounds[1].map((v, i) => v - p.bounds[0][i])) : 1;
    const scale = (Math.min(w, h) / extent) * state.zoom;
    const points = p.points.map((pt) => transform(pt, center, scale));
    const faces = p.faces.map((face, i) => ({ face, i, z: face.reduce((sum, n) => sum + points[n][2], 0) / 4 })).sort((a, b) => a.z - b.z);
    ctx.lineJoin = 'round';
    for (const item of faces) {
      const kind = p.kind[item.i]; if (kind === 0 && !$('show-boundary').checked) continue;
      ctx.beginPath(); item.face.forEach((n, j) => { const pt = points[n]; const x = w / 2 + pt[0], y = h / 2 - pt[1]; if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.closePath();
      if (kind === 1) { ctx.fillStyle = qualityColor(p.quality[item.i], .74); ctx.strokeStyle = 'rgba(17,35,31,.34)'; ctx.lineWidth = 0.65 * (window.devicePixelRatio || 1); }
      else { ctx.fillStyle = 'rgba(255,255,249,.035)'; ctx.strokeStyle = 'rgba(51,72,64,.24)'; ctx.lineWidth = 0.55 * (window.devicePixelRatio || 1); }
      ctx.fill(); ctx.stroke();
    }
  }

  canvas.addEventListener('pointerdown', (event) => { state.drag = [event.clientX, event.clientY, state.yaw, state.pitch]; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', (event) => { if (!state.drag) return; state.yaw = state.drag[2] + (event.clientX - state.drag[0]) * .008; state.pitch = Math.max(-1.45, Math.min(1.45, state.drag[3] + (event.clientY - state.drag[1]) * .008)); render(); });
  canvas.addEventListener('pointerup', () => { state.drag = null; });
  canvas.addEventListener('wheel', (event) => { event.preventDefault(); state.zoom = Math.max(.25, Math.min(2.5, state.zoom * Math.exp(-event.deltaY * .001))); render(); }, { passive: false });
  $('reset-view').addEventListener('click', () => { state.yaw = -.65; state.pitch = .45; state.zoom = .78; render(); });
  $('show-boundary').addEventListener('change', render);
  $('generate').addEventListener('click', generate);
  window.addEventListener('resize', resize); new ResizeObserver(resize).observe(canvas);

  fetch('/api/health').then((r) => r.json()).then((data) => { $('health-dot').classList.add('online'); $('health').textContent = `Local engine ${data.version}`; }).catch(() => { $('health').textContent = 'Engine unavailable'; });
  resize();
})();
