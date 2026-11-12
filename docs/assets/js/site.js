/* SPECTRA project page: results table, band-selection figure, small utilities.
   All numbers come from assets/data/paper.js, which is generated from the camera-ready LaTeX. */
(function () {
  'use strict';
  var P = window.SPECTRA_PAPER;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var el = function (tag, attrs, html) {
    var e = document.createElement(tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  };
  var NS = 'http://www.w3.org/2000/svg';
  var svg = function (tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  };

  /** A row of pill buttons acting as a single-choice control. */
  function segmented(host, options, value, onChange) {
    host.innerHTML = '';
    options.forEach(function (o) {
      var b = el('button', { type: 'button', 'aria-pressed': String(o.value === value) }, o.label);
      if (o.title) b.title = o.title;
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(host.children, function (c) { c.setAttribute('aria-pressed', 'false'); });
        b.setAttribute('aria-pressed', 'true');
        onChange(o.value);
      });
      host.appendChild(b);
    });
  }
  window.SPECTRA_UI = { segmented: segmented, el: el, svg: svg };

  function shortBand(name) {
    if (name === 'AUX') return 'AUX';
    var m = /^(B\d+A?)_/.exec(name);
    if (m) return m[1];
    return name.replace('NIR_NARROW', 'NIR').replace('SWIR_1', 'SWIR1').replace('SWIR_2', 'SWIR2')
      .replace('BLUE', 'Blue').replace('GREEN', 'Green').replace('RED', 'Red');
  }
  function longBand(b) {
    var nice = b.name.replace(/_/g, ' ');
    return b.center > 0 ? nice + ' · ' + b.center + ' nm (FWHM ' + b.fwhm + ' nm)' : nice + ' · auxiliary, no wavelength';
  }
  window.SPECTRA_UI.shortBand = shortBand;
  window.SPECTRA_UI.longBand = longBand;

  /* ---------------------------------------------------------------- main comparison table */
  var METHODS = ['LP', 'LoRA-16', 'LoRA-32', 'LoRA-64', 'Surgical', 'Full-FT', 'SPECTRA'];
  var DATASETS = ['Sen1Floods11', 'FireScars', 'Landslide4Sense', 'SA Crop Type'];
  var GEOFMS = ['Prithvi-EO-2.0 600M', 'ScaleMAE', 'SatMAE'];
  var METRICS = [
    { value: 'miou', label: 'Macro mIoU' }, { value: 'f1', label: 'Macro F1' }, { value: 'fg_iou', label: 'FG IoU' }
  ];
  function renderMain() {
    var host = $('#main-table');
    if (!host || !P) return;
    var state = { geofm: 'ScaleMAE', metric: 'miou' };
    function draw() {
      var rows = P.mainTable.filter(function (r) { return r.backbone === state.geofm; });
      var get = function (m, d) { return rows.filter(function (r) { return r.method === m && r.dataset === d; })[0]; };
      var best = {};
      DATASETS.forEach(function (d) {
        var top = -Infinity;
        METHODS.forEach(function (m) { if (m !== 'Full-FT') top = Math.max(top, get(m, d)[state.metric]); });
        best[d] = top;
      });
      var h = '<table class="ptable"><thead><tr><th>Method</th><th class="num">Trainable params</th>';
      DATASETS.forEach(function (d) { h += '<th class="num">' + d + '</th>'; });
      h += '</tr></thead><tbody>';
      METHODS.forEach(function (m) {
        var pcts = DATASETS.map(function (d) { return get(m, d).trainable_pct; });
        var lo = Math.min.apply(null, pcts), hi = Math.max.apply(null, pcts);
        var cls = m === 'SPECTRA' ? 'ours' : (m === 'Full-FT' ? 'ref' : '');
        h += '<tr class="' + cls + '"><td>' + m + (m === 'Full-FT' ? ' <small>high-cost reference</small>' : '') +
          (m === 'LP' ? ' <small>linear probing</small>' : '') + '</td>';
        h += '<td class="num">' + (lo === hi ? lo.toFixed(2) + '%' : lo.toFixed(2) + '&ndash;' + hi.toFixed(2) + '%') + '</td>';
        DATASETS.forEach(function (d) {
          var r = get(m, d), v = r[state.metric], sd = r[state.metric + '_sd'];
          var isBest = m !== 'Full-FT' && v === best[d];
          h += '<td class="num' + (isBest ? ' best' : '') + '">' + v.toFixed(2) + ' <span class="sd">&plusmn; ' + sd.toFixed(2) + '</span>' +
            (m === 'SPECTRA' && lo !== hi ? '<small>' + r.trainable_pct.toFixed(2) + '% trainable</small>' : '') + '</td>';
        });
        h += '</tr>';
      });
      h += '</tbody></table>';
      $('#main-table .table-wrap').innerHTML = h;
    }
    segmented($('#mt-geofm'), GEOFMS.map(function (g) { return { value: g, label: g }; }), state.geofm, function (v) { state.geofm = v; draw(); });
    segmented($('#mt-metric'), METRICS, state.metric, function (v) { state.metric = v; draw(); });
    draw();
  }

  /* ---------------------------------------------------------------- band-selection figure */
  var BB_SHORT = { 'Prithvi-EO-2.0': 'Prithvi', 'ScaleMAE': 'ScaleMAE', 'SatMAE': 'SatMAE' };
  function renderBandFig() {
    var host = $('#bandfig');
    if (!host || !P) return;
    var state = { geofm: 'ScaleMAE', dataset: 'Sen1Floods11', mode: 'sel' };
    var W = 960, H = 232, chipH = 34;
    function draw() {
      var tb = P.targetBands[state.dataset], sb = P.sourceBands[BB_SHORT[state.geofm]];
      var bs = P.bandSelection[BB_SHORT[state.geofm]][state.dataset];
      var C = tb.length, K = sb.length;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
        'aria-label': 'Band selection for ' + state.geofm + ' on ' + state.dataset + ': ' + K + ' of ' + C + ' bands kept.' });
      var gap = 6, cw = Math.min(62, (W - 40 - gap * (C - 1)) / C);
      var x0 = (W - (C * cw + (C - 1) * gap)) / 2;
      var kw = Math.min(74, (W - 200 - gap * (K - 1)) / K);
      var k0 = (W - (K * kw + (K - 1) * gap)) / 2;
      var yT = 30, yS = 168;
      var cx = function (i) { return x0 + i * (cw + gap) + cw / 2; };
      var kx = function (j) { return k0 + j * (kw + gap) + kw / 2; };
      s.appendChild(svg('text', { x: W / 2, y: 16, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 600, fill: '#363636' },
        state.dataset + ' provides C = ' + C + ' bands'));
      s.appendChild(svg('text', { x: W / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 600, fill: '#363636' },
        state.geofm + ' patch embedding expects K = ' + K + ' bands'));
      var links = svg('g', { fill: 'none' });
      s.appendChild(links);
      if (state.mode === 'bre') {
        for (var i = 0; i < C; i++) for (var j = 0; j < K; j++) {
          links.appendChild(svg('line', { x1: cx(i), y1: yT + chipH, x2: kx(j), y2: yS, stroke: '#eb6834', 'stroke-width': 1, opacity: .28 }));
        }
      }
      bs.selected.forEach(function (ti, j) {
        links.appendChild(svg('line', { x1: cx(ti), y1: yT + chipH, x2: kx(j), y2: yS, stroke: '#2a78d6', 'stroke-width': 2.25 }));
      });
      tb.forEach(function (b, i) {
        var kept = bs.selected.indexOf(i) >= 0;
        var g = svg('g', {});
        g.appendChild(svg('title', {}, 'Band ' + i + ': ' + longBand(b) + (kept ? ' — kept by band selection' : ' — discarded by band selection')));
        g.appendChild(svg('rect', { x: cx(i) - cw / 2, y: yT, width: cw, height: chipH, rx: 6,
          fill: kept ? '#2a78d6' : '#ffffff', stroke: kept ? '#2a78d6' : (state.mode === 'bre' ? '#eb6834' : '#a9a9a4'),
          'stroke-width': 1.5, 'stroke-dasharray': kept ? '' : '4 3' }));
        g.appendChild(svg('text', { x: cx(i), y: yT + 15, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 600,
          fill: kept ? '#fff' : '#52514e' }, shortBand(b.name)));
        g.appendChild(svg('text', { x: cx(i), y: yT + 28, 'text-anchor': 'middle', 'font-size': 9.5,
          fill: kept ? '#e8f1fc' : '#8a8985' }, b.center > 0 ? Math.round(b.center) + ' nm' : 'aux'));
        s.appendChild(g);
      });
      sb.forEach(function (b, j) {
        var g = svg('g', {});
        g.appendChild(svg('title', {}, 'Pretrained slot ' + j + ': ' + longBand(b)));
        g.appendChild(svg('rect', { x: kx(j) - kw / 2, y: yS, width: kw, height: chipH, rx: 6, fill: '#363636' }));
        g.appendChild(svg('text', { x: kx(j), y: yS + 15, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 600, fill: '#fff' }, shortBand(b.name)));
        g.appendChild(svg('text', { x: kx(j), y: yS + 28, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#d6d6d6' }, Math.round(b.center) + ' nm'));
        s.appendChild(g);
      });
      var holder = $('#bandfig .scroll');
      holder.innerHTML = '';
      holder.appendChild(s);
      var nKept = bs.selected.filter(function (v, i, a) { return a.indexOf(v) === i; }).length;
      var disc = bs.discarded.map(function (i) { return shortBand(tb[i].name); });
      var msg;
      if (!disc.length) {
        msg = 'Every band is already used: <b>nothing is discarded</b>' +
          (nKept < K ? ', and ' + (K - nKept) + ' pretrained slots are filled by repeating a band' : '') +
          '. This is the case where extra-band routing has little to add.';
      } else if (state.mode === 'sel') {
        msg = 'Band selection keeps <b>' + nKept + ' of ' + C + '</b> bands and discards <b>' + disc.length + '</b>: ' + disc.join(', ') + '.';
      } else {
        msg = 'BRE keeps the same ' + nKept + '-band anchor and adds a learned residual in which <b>all ' + C + ' bands</b> can reach every pretrained slot.';
      }
      $('#bandfig .summary').innerHTML = msg +
        ' <span style="color:#6b6b6b">Transfer gap of the frozen encoder with band selection: ' + bs.gap.toFixed(3) + ' (lower means better aligned).</span>';
    }
    segmented($('#bf-geofm'), Object.keys(BB_SHORT).map(function (g) { return { value: g, label: g }; }), state.geofm, function (v) { state.geofm = v; draw(); });
    segmented($('#bf-dataset'), DATASETS.map(function (d) { return { value: d, label: d }; }), state.dataset, function (v) { state.dataset = v; draw(); });
    segmented($('#bf-mode'), [{ value: 'sel', label: 'Band selection' }, { value: 'bre', label: 'BRE' }], state.mode, function (v) { state.mode = v; draw(); });
    draw();
  }

  /* ---------------------------------------------------------------- utilities */
  function wireCopy() {
    var b = $('#copy-bib');
    if (!b) return;
    b.addEventListener('click', function () {
      var text = $('#bibtex').textContent;
      var done = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, function () {});
      else { var r = document.createRange(); r.selectNodeContents($('#bibtex')); var s = getSelection(); s.removeAllRanges(); s.addRange(r); }
    });
  }
  function math() {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, { delimiters: [{ left: '\\(', right: '\\)', display: false }, { left: '$$', right: '$$', display: true }] });
    }
  }
  document.addEventListener('DOMContentLoaded', function () {
    renderMain(); renderBandFig(); wireCopy(); math();
  });
})();
