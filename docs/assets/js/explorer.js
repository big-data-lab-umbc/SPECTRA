/* SPECTRA inference explorer.
   Shows precomputed outputs of the paper's seed-42 checkpoints on held-out test tiles.
   Data: assets/demo/manifest.js (+ per-tile band and mask sprites), written by tools/export_demo.py. */
(function () {
  'use strict';
  var D = window.SPECTRA_DEMO, P = window.SPECTRA_PAPER, UI = window.SPECTRA_UI;
  var root = document.getElementById('explorer');
  if (!root) return;
  if (!D || !P || !UI) { root.innerHTML = '<p class="ex-error">The explorer data did not load.</p>'; return; }

  var BASE = 'assets/demo/';
  /* The single-file review build inlines the demo images as data URIs in window.SPECTRA_ASSETS. */
  function asset(p) { var a = window.SPECTRA_ASSETS; return a && a[p] ? a[p] : BASE + p; }
  var DS = ['sen1floods11', 'fire_scars', 'landslide4sense', 'geobench_sa_crop_type'];
  var BB = ['prithvi', 'scalemae', 'satmae'];
  var MK = ['lora32', 'surgical', 'spectra', 'fullft'];
  var BB_TABLE = { prithvi: 'Prithvi-EO-2.0 600M', scalemae: 'ScaleMAE', satmae: 'SatMAE' };
  var BB_SEL = { prithvi: 'Prithvi', scalemae: 'ScaleMAE', satmae: 'SatMAE' };
  var BB_LABEL = { prithvi: 'Prithvi-EO-2.0', scalemae: 'ScaleMAE', satmae: 'SatMAE' };
  var M_LABEL = { lora32: 'LoRA-32', surgical: 'Surgical', spectra: 'SPECTRA', fullft: 'Full-FT' };
  var CLASSES = {
    sen1floods11: ['Not water', 'Water'],
    fire_scars: ['Unburned', 'Burn scar'],
    landslide4sense: ['No landslide', 'Landslide'],
    geobench_sa_crop_type: ['No data', 'Lucerne/Medics', 'Planted pastures (perennial)', 'Fallow', 'Wine grapes', 'Weeds',
      'Small grain grazing', 'Wheat', 'Canola', 'Rooibos']
  };
  var TASK = { sen1floods11: 'Flood mapping', fire_scars: 'Burn-scar mapping', landslide4sense: 'Landslide mapping', geobench_sa_crop_type: 'Crop-type mapping (10 classes)' };
  /* band indices for the extra composites: [R, G, B] channels */
  var COMPOSITE = {
    sen1floods11: { cir: [7, 3, 2], swir: [11, 7, 3] },
    fire_scars: { cir: [3, 2, 1], swir: [5, 3, 2] },
    landslide4sense: { cir: [7, 3, 2], swir: [11, 7, 3] },
    geobench_sa_crop_type: { cir: [7, 3, 2], swir: [10, 7, 3] }
  };
  /* paper's convention for binary error maps */
  var TP = [0, 255, 0], FP = [0, 0, 255], FN = [255, 0, 0];
  /* class colours for the 10-class crop map: neutral for "No data", then the categorical order */
  var CLASS_RGB = [[208, 207, 201], [42, 120, 214], [235, 104, 52], [27, 175, 122], [237, 161, 0], [232, 123, 164],
    [0, 131, 0], [98, 80, 214], [227, 73, 72], [138, 90, 43]];
  var ERR_RGB = [226, 0, 116];

  var state = { ds: 'sen1floods11', bb: 'scalemae', tile: 0, view: 'rgb', band: null, overlay: 'errors', alpha: 0.75, showLabel: false };

  /* ------------------------------------------------------------ DOM skeleton */
  root.innerHTML =
    '<div class="ctl-row">' +
    '  <div class="ctl"><span>Dataset</span><div class="seg" id="ex-ds"></div></div>' +
    '  <div class="ctl"><span>GeoFM</span><div class="seg" id="ex-bb"></div></div>' +
    '</div>' +
    '<p class="ex-tilehead" id="ex-tilehead"></p>' +
    '<div class="ex-tiles" id="ex-tiles" role="group" aria-label="Test tiles"></div>' +
    '<div class="ex-main">' +
    '  <div class="ex-input">' +
    '    <div class="ex-head"><b>Input</b><span id="ex-viewname"></span></div>' +
    '    <div class="ex-canvas"><canvas id="ex-in" aria-label="Input tile"></canvas></div>' +
    '    <div class="seg small" id="ex-view"></div>' +
    '    <div class="ex-bands" id="ex-bands" role="group" aria-label="Input bands"></div>' +
    '    <p class="ex-note" id="ex-bandnote"></p>' +
    '  </div>' +
    '  <div>' +
    '    <div class="ex-head"><b>Predictions</b><span id="ex-predname"></span></div>' +
    '    <div class="ex-preds" id="ex-preds"></div>' +
    '  </div>' +
    '</div>' +
    '<div class="ex-under">' +
    '  <div class="ctl"><span>Overlay</span><div class="seg small" id="ex-ov"></div>' +
    '    <label style="font-size:.84rem;color:#6b6b6b">opacity <input type="range" id="ex-alpha" min="0.2" max="1" step="0.05" value="0.75" aria-label="Overlay opacity"></label>' +
    '    <label style="font-size:.84rem;color:#6b6b6b"><input type="checkbox" id="ex-gt"> show label on input</label></div>' +
    '  <div class="ex-legend" id="ex-legend"></div>' +
    '</div>' +
    '<p class="ex-status" id="ex-status" aria-live="polite"></p>' +
    '<div class="ex-run" id="ex-run"></div>';
  var $ = function (id) { return document.getElementById(id); };
  var predCanvases = {};
  MK.forEach(function (mk) {
    var f = UI.el('figure', { 'class': 'ex-pred fig' + (mk === 'spectra' ? ' ours' : '') },
      '<div class="ex-mhead"></div><div class="ex-canvas"><canvas aria-label="' + M_LABEL[mk] + ' prediction"></canvas></div><figcaption></figcaption>');
    $('ex-preds').appendChild(f);
    predCanvases[mk] = f;
  });

  /* ------------------------------------------------------------ pixel loading */
  var cache = {};
  function loadPixels(url) {
    return new Promise(function (resolve, reject) {
      var im = new Image();
      im.onload = function () {
        var c = document.createElement('canvas');
        c.width = im.naturalWidth; c.height = im.naturalHeight;
        var x = c.getContext('2d', { willReadFrequently: true });
        x.drawImage(im, 0, 0);
        try { resolve(x.getImageData(0, 0, c.width, c.height)); } catch (e) { reject(e); }
      };
      im.onerror = function () { reject(new Error('could not load ' + url)); };
      im.src = url;
    });
  }
  function getTile(ds, t) {
    var k = ds + '/' + t;
    if (!cache[k]) {
      var dir = ds + '/t' + t + '/';
      cache[k] = Promise.all([loadPixels(asset(dir + 'bands.jpg')), loadPixels(asset(dir + 'masks.png'))])
        .then(function (a) { return { bands: a[0], masks: a[1], bre: {} }; });
    }
    return cache[k];
  }
  function getBre(ds, t, bb, tile) {
    if (!tile.bre[bb]) tile.bre[bb] = loadPixels(asset(ds + '/t' + t + '/bre_' + bb + '.jpg'));
    return tile.bre[bb];
  }

  /* ------------------------------------------------------------ rendering */
  function info() { return D.datasets[state.ds]; }
  function cell() { return info().cells[state.bb]; }
  function hasBre() { return !!(info().tiles[state.tile].bre || {})[state.bb]; }
  function maskCol(bb, mk) { return 1 + BB.indexOf(bb) * MK.length + MK.indexOf(mk); }

  /** RGBA base image for the current tile: idx = [r,g,b] band indices, or a single band, or the BRE sprite. */
  function composite(tile, idx) {
    var S = info().size, C = info().n_bands, W = C * S, src = tile.bands.data, out = new Uint8ClampedArray(S * S * 4);
    for (var y = 0; y < S; y++) for (var x = 0; x < S; x++) {
      var o = (y * S + x) * 4, row = y * W;
      out[o] = src[(row + idx[0] * S + x) * 4];
      out[o + 1] = src[(row + idx[1] * S + x) * 4];
      out[o + 2] = src[(row + idx[2] * S + x) * 4];
      out[o + 3] = 255;
    }
    return out;
  }
  function fromBre(sprite, half) {  // half 0 = virtual image, 1 = |delta| magnitude
    var S = info().size, W = sprite.width, src = sprite.data, out = new Uint8ClampedArray(S * S * 4);
    for (var y = 0; y < S; y++) for (var x = 0; x < S; x++) {
      var o = (y * S + x) * 4, s = (y * W + half * S + x) * 4;
      if (half === 0) { out[o] = src[s]; out[o + 1] = src[s + 1]; out[o + 2] = src[s + 2]; }
      else {            // sequential single-hue ramp, light = little change
        var v = src[s] / 255;
        out[o] = 240 - v * (240 - 13); out[o + 1] = 239 - v * (239 - 54); out[o + 2] = 236 - v * (236 - 107);
      }
      out[o + 3] = 255;
    }
    return out;
  }
  function blend(px, o, rgb, a) {
    px[o] = px[o] * (1 - a) + rgb[0] * a; px[o + 1] = px[o + 1] * (1 - a) + rgb[1] * a; px[o + 2] = px[o + 2] * (1 - a) + rgb[2] * a;
  }
  /** Draw base + overlay. col = mask column of a prediction, or 0 for the label itself. */
  function paint(canvas, base, tile, col, mode, box) {
    var S = info().size, W = tile.masks.width, m = tile.masks.data, px = new Uint8ClampedArray(base), a = state.alpha;
    var binary = info().n_classes === 2;
    if (mode !== 'none') {
      for (var y = 0; y < S; y++) for (var x = 0; x < S; x++) {
        var o = (y * S + x) * 4, lab = m[(y * W + x) * 4], pr = m[(y * W + col * S + x) * 4];
        if (col !== 0 && pr === 254) { blend(px, o, [245, 245, 245], 0.72); continue; }     // outside the model's own crop
        if (lab === 255) { if (col !== 0) blend(px, o, [40, 40, 40], 0.55); continue; }   // unlabeled pixel
        if (col === 0) {                                    // ground truth
          if (binary) { if (lab === 1) blend(px, o, TP, a); } else blend(px, o, CLASS_RGB[lab], a);
        } else if (mode === 'errors') {
          if (binary) {
            if (lab === 1 && pr === 1) blend(px, o, TP, a);
            else if (lab === 0 && pr === 1) blend(px, o, FP, a);
            else if (lab === 1 && pr === 0) blend(px, o, FN, a);
          } else if (pr !== lab) blend(px, o, ERR_RGB, a);
        } else if (mode === 'pred') {                       // prediction only
          if (binary) { if (pr === 1) blend(px, o, TP, a); } else blend(px, o, CLASS_RGB[pr], a);
        }
      }
    }
    canvas.width = S; canvas.height = S;
    var ctx = canvas.getContext('2d');
    ctx.putImageData(new ImageData(px, S, S), 0, 0);
    if (box && col !== 0) { ctx.strokeStyle = '#1f1f1f'; ctx.lineWidth = 1; ctx.strokeRect(box[0] - 0.5, box[1] - 0.5, box[2] + 1, box[3] + 1); }
  }

  var fmt = function (v) { return v == null ? 'n/a' : v.toFixed(3); };
  var signed = function (v) { return (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3); };
  function paperRow(mk) {
    var bbL = BB_TABLE[state.bb], dsL = info().label, mL = M_LABEL[mk];
    return P.mainTable.filter(function (r) { return r.backbone === bbL && r.dataset === dsL && r.method === mL; })[0];
  }

  var token = 0;
  function render() {
    var my = ++token, I = info(), t = state.tile, T = I.tiles[t];
    if (state.view === 'bre' || state.view === 'delta') { if (!hasBre()) state.view = 'rgb'; }
    getTile(state.ds, t).then(function (tile) {
      if (my !== token) return;
      var rgb = composite(tile, I.rgb);
      var pIn;
      if (state.view === 'bre' || state.view === 'delta') {
        pIn = getBre(state.ds, t, state.bb, tile).then(function (s) { return fromBre(s, state.view === 'bre' ? 0 : 1); });
      } else if (state.view === 'band') pIn = Promise.resolve(composite(tile, [state.band, state.band, state.band]));
      else if (state.view === 'rgb') pIn = Promise.resolve(rgb);
      else pIn = Promise.resolve(composite(tile, COMPOSITE[state.ds][state.view]));
      pIn.then(function (base) {
        if (my !== token) return;
        paint($('ex-in'), base, tile, 0, state.showLabel ? 'label' : 'none');
      });
      MK.forEach(function (mk) {
        var fig = predCanvases[mk], rec = cell().methods[mk], tm = rec.tile_metrics[t], pr = paperRow(mk);
        paint(fig.querySelector('canvas'), rgb, tile, maskCol(state.bb, mk), state.overlay === 'none' && cell().box ? 'pred0' : state.overlay, cell().box);
        var second = I.n_classes === 2 ? 'FG IoU ' + fmt(tm.fg_iou) : 'acc. ' + fmt(tm.pixel_acc);
        var gain = '';
        if (mk === 'spectra') {
          var g = tm.miou - cell().methods.lora32.tile_metrics[t].miou;
          gain = '<span class="gain' + (g < 0 ? ' neg' : '') + '" title="SPECTRA minus LoRA-32, per-tile mIoU">' + signed(g) + '</span>';
        }
        fig.querySelector('.ex-mhead').innerHTML = '<b>' + M_LABEL[mk] + '</b><span class="mv">mIoU ' + fmt(tm.miou) + gain + ' · ' + second + '</span>';
        fig.querySelector('figcaption').innerHTML = '<span class="sub">' + (mk === 'spectra' ? 'All bands (BRE)' : 'Band selection') + ' · ' +
          pr.trainable_pct.toFixed(2) + '% trainable' + (cell().box ? ' · metrics inside the outlined ' + cell().box[2] + '-px crop' : '') + '</span>' +
          '<span class="p">Test set, 3 seeds (paper): ' + pr.miou.toFixed(2) + ' ± ' + pr.miou_sd.toFixed(2) + ' mIoU</span>';
      });
      root.dataset.ready = '1';
    }).catch(function (e) {
      $('ex-status').textContent = location.protocol === 'file:'
        ? 'The explorer needs the page to be served over http(s); it cannot read image pixels from file://.'
        : 'Could not load this tile (' + e.message + ').';
    });
    $('ex-predname').textContent = BB_LABEL[state.bb] + ' · ' + I.label + ' · ' + T.name + (T.in_paper ? ' (in the paper’s Fig. 4)' : '');
    var names = { rgb: 'True colour', cir: 'Colour infrared (NIR, red, green)', swir: 'SWIR composite (SWIR, NIR, red)',
      bre: 'Virtual image the ' + BB_LABEL[state.bb] + ' patch embedding receives', delta: 'Mean |Δ| the residual path adds' };
    $('ex-viewname').textContent = state.view === 'band' ? 'Band ' + state.band + ': ' + UI.longBand(P.targetBands[I.label][state.band]) : names[state.view];
  }

  /* ------------------------------------------------------------ controls */
  function buildTiles() {
    var host = $('ex-tiles'), I = info(), M = cell().methods;
    host.innerHTML = '';
    $('ex-tilehead').innerHTML = 'Test tiles &middot; under each: SPECTRA&rsquo;s per-tile mIoU gain over LoRA-32 with ' + BB_LABEL[state.bb];
    I.tiles.forEach(function (T, i) {
      var g = M.spectra.tile_metrics[i].miou - M.lora32.tile_metrics[i].miou;
      var wrap = UI.el('div', { 'class': 'ex-tilewrap' });
      var b = UI.el('button', { type: 'button', 'class': 'ex-tile', 'aria-pressed': String(i === state.tile), title: T.name,
        'aria-label': 'Tile ' + T.name + ', SPECTRA gain ' + signed(g) });
      b.style.backgroundImage = 'url(' + asset(state.ds + '/thumbs.jpg') + ')';
      b.style.backgroundSize = (I.tiles.length * 100) + '% 100%';
      b.style.backgroundPosition = (I.tiles.length > 1 ? (i / (I.tiles.length - 1)) * 100 : 0) + '% 0';
      if (T.in_paper) b.appendChild(UI.el('span', { 'class': 'tag' }, 'Fig. 4'));
      b.addEventListener('click', function () {
        state.tile = i;
        Array.prototype.forEach.call(host.querySelectorAll('.ex-tile'), function (c) { c.setAttribute('aria-pressed', 'false'); });
        b.setAttribute('aria-pressed', 'true');
        buildViews(); render();
      });
      wrap.appendChild(b);
      wrap.appendChild(UI.el('span', { 'class': 'ex-tgain' + (g < 0 ? ' neg' : '') }, signed(g)));
      host.appendChild(wrap);
    });
  }
  function buildViews() {
    var opts = [{ value: 'rgb', label: 'True colour' }, { value: 'cir', label: 'Infrared' }, { value: 'swir', label: 'SWIR' }];
    if (hasBre()) {
      opts.push({ value: 'bre', label: 'Virtual image', title: 'The K-band image SPECTRA feeds to the frozen patch embedding, shown in its red/green/blue slots' });
      opts.push({ value: 'delta', label: 'Residual', title: 'How much the all-band residual path changes the selected-band input' });
    } else if (state.view === 'bre' || state.view === 'delta') state.view = 'rgb';
    UI.segmented($('ex-view'), opts, state.view === 'band' ? null : state.view, function (v) { state.view = v; state.band = null; buildBands(); render(); });
  }
  function buildBands() {
    var host = $('ex-bands'), I = info(), tb = P.targetBands[I.label], bs = P.bandSelection[BB_SEL[state.bb]][I.label];
    host.innerHTML = '';
    tb.forEach(function (b, i) {
      var kept = bs.selected.indexOf(i) >= 0;
      var c = UI.el('button', { type: 'button', 'class': 'chip' + (kept ? '' : ' off'), 'aria-pressed': String(state.view === 'band' && state.band === i),
        title: UI.longBand(b) + (kept ? ' — kept by band selection' : ' — discarded by band selection') }, UI.shortBand(b.name));
      c.addEventListener('click', function () { state.view = 'band'; state.band = i; buildViews(); buildBands(); render(); });
      host.appendChild(c);
    });
    var nKept = bs.selected.filter(function (v, i, a) { return a.indexOf(v) === i; }).length;
    $('ex-bandnote').innerHTML = 'Click a band to view it. For ' + BB_LABEL[state.bb] + ', band selection keeps the <b style="color:#2a78d6">' + nKept +
      ' filled</b> band' + (nKept > 1 ? 's' : '') + (bs.discarded.length ? ' and discards the ' + bs.discarded.length + ' dashed ones; SPECTRA’s BRE reads all ' + tb.length + '.'
        : '; none are discarded for this pair.');
  }
  function buildLegend() {
    var host = $('ex-legend'), I = info(), h = '';
    var sw = function (rgb, label) { return '<span><i style="background:rgb(' + rgb.join(',') + ')"></i>' + label + '</span>'; };
    if (I.n_classes === 2) {
      var fg = CLASSES[state.ds][1].toLowerCase();
      if (state.overlay === 'errors') h = sw(TP, 'Correct ' + fg + ' (TP)') + sw(FP, 'False alarm (FP)') + sw(FN, 'Missed (FN)');
      else if (state.overlay === 'pred') h = sw(TP, 'Predicted ' + fg);
    } else if (state.overlay === 'errors') h = sw(ERR_RGB, 'Pixel classified wrongly');
    else if (state.overlay === 'pred') CLASSES[state.ds].forEach(function (n, i) { h += sw(CLASS_RGB[i], n); });
    host.innerHTML = h;
  }
  function buildRun() {
    var host = $('ex-run'), I = info(), sp = cell().methods.spectra, pr = paperRow('spectra'), lo = paperRow('lora32');
    var ranks = sp.ranks || [];
    var W = 300, H = 166, x0 = 34, y0 = 134, top = 34, bw = 34, step = (W - x0 - 12) / 4;
    var ymax = Math.max(64, Math.max.apply(null, ranks.concat([32])));
    var yy = function (v) { return y0 - (v / ymax) * (y0 - top); };
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="LoRA rank per encoder stage: ' + ranks.join(', ') + '">';
    [0, 32, 64].forEach(function (v) {
      s += '<line x1="' + x0 + '" x2="' + (W - 8) + '" y1="' + yy(v) + '" y2="' + yy(v) + '" stroke="' + (v === 0 ? '#b9b8b2' : '#ececec') + '" stroke-width="1"/>' +
        '<text x="' + (x0 - 6) + '" y="' + (yy(v) + 4) + '" text-anchor="end" font-size="10" fill="#6b6b6b">' + v + '</text>';
    });
    s += '<line x1="' + x0 + '" x2="' + (W - 8) + '" y1="' + yy(32) + '" y2="' + yy(32) + '" stroke="#6b6b6b" stroke-width="1.25" stroke-dasharray="5 4"/>' +
      '<line x1="' + (W - 112) + '" x2="' + (W - 90) + '" y1="10" y2="10" stroke="#6b6b6b" stroke-width="1.25" stroke-dasharray="5 4"/>' +
      '<text x="' + (W - 8) + '" y="13.5" text-anchor="end" font-size="10" fill="#6b6b6b">uniform LoRA-32</text>';
    ranks.forEach(function (r, i) {
      var cx = x0 + step * i + step / 2, h = Math.max(1, y0 - yy(r));
      s += '<path d="M' + (cx - bw / 2) + ' ' + y0 + ' v' + (-(h - 4 > 0 ? h - 4 : 0)) + ' q0 -4 4 -4 h' + (bw - 8) + ' q4 0 4 4 v' + (h - 4 > 0 ? h - 4 : 0) + ' z" fill="#eb6834">' +
        '<title>Stage ' + (i + 1) + ': rank ' + r + '</title></path>' +
        '<text x="' + cx + '" y="' + (y0 - h - 5) + '" text-anchor="middle" font-size="11" font-weight="600" fill="#363636">' + r + '</text>' +
        '<text x="' + cx + '" y="' + (y0 + 15) + '" text-anchor="middle" font-size="10.5" fill="#6b6b6b">Stage ' + (i + 1) + '</text>';
    });
    s += '</svg>';
    var sum = ranks.reduce(function (a, b) { return a + b; }, 0);
    var left = '<div><h4>Stage-wise LoRA ranks in this SPECTRA run</h4>' +
      '<p>Total rank ' + sum + ' against 128 for uniform LoRA-32. Trainable parameters, test-set average in the paper: ' +
      pr.trainable_pct.toFixed(2) + '% for SPECTRA, ' + lo.trainable_pct.toFixed(2) + '% for LoRA-32.</p>' + s + '</div>';

    var right = '<div><h4>Learned band gates</h4>';
    if (sp.gates) {
      var G = sp.gates, K = G.length, C = G[0].length, tb = P.targetBands[I.label], sb = P.sourceBands[BB_SEL[state.bb]];
      var d = 0; G.forEach(function (r) { r.forEach(function (v) { d = Math.max(d, Math.abs(v - 1)); }); });
      d = Math.max(0.02, Math.ceil(d * 100) / 100);
      var cw = 36, ch = K > 6 ? 17 : 22, lx = 58, ty = 20, GW = lx + C * cw + 6, GH = ty + K * ch + 34;
      var col = function (v) {           // diverging: blue below 1, neutral at 1, red above 1
        var t = Math.max(-1, Math.min(1, (v - 1) / d)), pole = t < 0 ? [42, 120, 214] : [227, 73, 72], mid = [240, 239, 236], a = Math.abs(t);
        return 'rgb(' + mid.map(function (m, i) { return Math.round(m + (pole[i] - m) * a); }).join(',') + ')';
      };
      var g = '<div class="scroll"><svg viewBox="0 0 ' + GW + ' ' + GH + '" style="min-width:' + Math.min(GW, 420) + 'px" role="img" aria-label="Gate table, ' + K + ' virtual bands by ' + C + ' input bands">';
      tb.forEach(function (b, i) {
        g += '<text x="' + (lx + i * cw + cw / 2) + '" y="' + (ty - 6) + '" text-anchor="middle" font-size="9.5" fill="#6b6b6b">' + UI.shortBand(b.name) + '</text>';
      });
      G.forEach(function (row, j) {
        g += '<text x="' + (lx - 6) + '" y="' + (ty + j * ch + ch / 2 + 3.5) + '" text-anchor="end" font-size="9.5" fill="#6b6b6b">' + UI.shortBand(sb[j].name) + '</text>';
        row.forEach(function (v, i) {
          var strong = Math.abs(v - 1) / d > 0.62;
          g += '<rect x="' + (lx + i * cw + 1) + '" y="' + (ty + j * ch + 1) + '" width="' + (cw - 2) + '" height="' + (ch - 2) + '" rx="2" fill="' + col(v) + '">' +
            '<title>Input ' + UI.shortBand(tb[i].name) + ' → virtual ' + UI.shortBand(sb[j].name) + ': gate ' + v.toFixed(3) + '</title></rect>' +
            '<text x="' + (lx + i * cw + cw / 2) + '" y="' + (ty + j * ch + ch / 2 + 3.2) + '" text-anchor="middle" font-size="8.5" fill="' + (strong ? '#fff' : '#363636') + '" pointer-events="none">' + v.toFixed(2) + '</text>';
        });
      });
      var ly = ty + K * ch + 12, lw = 150;
      g += '<defs><linearGradient id="gatescale"><stop offset="0" stop-color="rgb(42,120,214)"/><stop offset=".5" stop-color="rgb(240,239,236)"/><stop offset="1" stop-color="rgb(227,73,72)"/></linearGradient></defs>' +
        '<rect x="' + lx + '" y="' + ly + '" width="' + lw + '" height="8" rx="2" fill="url(#gatescale)"/>' +
        '<text x="' + lx + '" y="' + (ly + 20) + '" font-size="9.5" fill="#6b6b6b">' + (1 - d).toFixed(2) + '</text>' +
        '<text x="' + (lx + lw / 2) + '" y="' + (ly + 20) + '" text-anchor="middle" font-size="9.5" fill="#6b6b6b">1.00 (initial)</text>' +
        '<text x="' + (lx + lw) + '" y="' + (ly + 20) + '" text-anchor="end" font-size="9.5" fill="#6b6b6b">' + (1 + d).toFixed(2) + '</text></svg></div>';
      right += '<p>Rows are the ' + K + ' virtual bands, columns the ' + C + ' input bands. Every gate starts at 1; in this checkpoint the largest change after training is ' +
        d.toFixed(2) + '. Hover a cell for its value.</p>' + g;
    } else if (sp.has_adapter) {
      right += '<p>This run’s input adapter has a residual path but no gate table, so there is nothing to show here. Use “Virtual image” above to see what it feeds the patch embedding.</p>';
    } else {
      right += '<p>' + BB_LABEL[state.bb] + ' on ' + I.label + ' is a native band match: all ' + I.n_bands + ' bands already go to the patch embedding, so SPECTRA skips BRE here and only the stage-wise LoRA is active.</p>';
    }
    host.innerHTML = left + right + '</div>';
  }

  function hover(canvas, col, name) {
    canvas.addEventListener('mousemove', function (ev) {
      var I = info(), S = I.size, r = canvas.getBoundingClientRect();
      var x = Math.floor((ev.clientX - r.left) / r.width * S), y = Math.floor((ev.clientY - r.top) / r.height * S);
      if (x < 0 || y < 0 || x >= S || y >= S) return;
      getTile(state.ds, state.tile).then(function (tile) {
        var m = tile.masks.data, W = tile.masks.width, lab = m[(y * W + x) * 4], names = CLASSES[state.ds];
        var txt = 'Pixel (' + x + ', ' + y + ') · label: ' + (lab === 255 ? 'unlabeled' : names[lab]);
        var c = col();
        if (c) { var pv = m[(y * W + c * S + x) * 4]; txt += ' · ' + name() + ': ' + (pv === 254 ? 'outside this model’s crop' : names[pv]); }
        $('ex-status').textContent = txt;
      });
    });
    canvas.addEventListener('mouseleave', function () { $('ex-status').textContent = ''; });
  }

  UI.segmented($('ex-ds'), DS.map(function (d) { return { value: d, label: D.datasets[d].label, title: TASK[d] }; }), state.ds, function (v) {
    state.ds = v; state.tile = 0; state.view = 'rgb'; state.band = null;
    if (D.datasets[v].n_classes !== 2 && state.overlay === 'errors') state.overlay = 'pred';
    buildOverlay(); buildTiles(); buildViews(); buildBands(); buildLegend(); buildRun(); render();
  });
  UI.segmented($('ex-bb'), BB.map(function (b) { return { value: b, label: BB_LABEL[b] }; }), state.bb, function (v) {
    state.bb = v; buildTiles(); buildViews(); buildBands(); buildRun(); render();
  });
  function buildOverlay() {
    UI.segmented($('ex-ov'), [{ value: 'errors', label: 'Errors' }, { value: 'pred', label: 'Prediction' }, { value: 'none', label: 'Off' }], state.overlay,
      function (v) { state.overlay = v; buildLegend(); render(); });
  }
  $('ex-alpha').addEventListener('input', function (e) { state.alpha = parseFloat(e.target.value); render(); });
  $('ex-gt').addEventListener('change', function (e) { state.showLabel = e.target.checked; render(); });
  hover($('ex-in'), function () { return 0; }, function () { return ''; });
  MK.forEach(function (mk) {
    hover(predCanvases[mk].querySelector('canvas'), function () { return maskCol(state.bb, mk); }, function () { return M_LABEL[mk]; });
  });
  buildOverlay(); buildTiles(); buildViews(); buildBands(); buildLegend(); buildRun(); render();
})();
