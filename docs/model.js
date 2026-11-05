// ActivityCNN inference in plain JavaScript (same maths as src/depresjon_cnn/cnn.py)
const CNN = (() => {
  // x: array of C channels, each Float64Array(L). W: (C*k) x O stored as nested arrays [row][o]
  function conv1d(x, W, b, k) {
    const C = x.length, L = x[0].length, O = b.length, left = Math.floor((k - 1) / 2);
    const out = Array.from({ length: O }, () => new Float64Array(L));
    for (let t = 0; t < L; t++) {
      for (let o = 0; o < O; o++) {
        let s = b[o];
        for (let c = 0; c < C; c++) {
          const xc = x[c];
          for (let j = 0; j < k; j++) {
            const idx = t + j - left;
            if (idx >= 0 && idx < L) s += xc[idx] * W[c * k + j][o];
          }
        }
        out[o][t] = s > 0 ? s : 0; // ReLU fused
      }
    }
    return out;
  }
  function maxpool(x, size) {
    return x.map(ch => {
      const n = Math.floor(ch.length / size), out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let m = -Infinity;
        for (let j = 0; j < size; j++) m = Math.max(m, ch[i * size + j]);
        out[i] = m;
      }
      return out;
    });
  }
  function predictDay(binned, Wt) {
    // binned: 288 values of log1p activity averaged per 5 minutes
    let x = [Float64Array.from(binned, v => (v - Wt.mu) / Wt.sd)];
    x = maxpool(conv1d(x, Wt['0_W'], Wt['0_b'], 9), 4);
    x = maxpool(conv1d(x, Wt['3_W'], Wt['3_b'], 5), 4);
    x = conv1d(x, Wt['6_W'], Wt['6_b'], 3);
    const gap = x.map(ch => ch.reduce((a, v) => a + v, 0) / ch.length);
    let z = Wt['10_b'][0];
    for (let i = 0; i < gap.length; i++) z += gap[i] * Wt['10_W'][i][0];
    return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
  }
  // Raw minute counts (1440) -> 288 bins of log1p
  function binDay(minutes) {
    const out = new Array(288);
    for (let i = 0; i < 288; i++) {
      let s = 0;
      for (let j = 0; j < 5; j++) s += Math.log1p(minutes[i * 5 + j]);
      out[i] = s / 5;
    }
    return out;
  }
  // CSV text with columns timestamp, activity -> [{date, minutes[1440]}] (only complete days)
  function parseCSV(text) {
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    if (!lines.length) throw new Error('The file is empty.');
    const head = lines[0].toLowerCase().split(/[,;\t]/).map(s => s.trim().replace(/"/g, ''));
    const ti = head.indexOf('timestamp'), ai = head.indexOf('activity');
    if (ti < 0 || ai < 0) throw new Error('The file needs a header row with "timestamp" and "activity" columns.');
    const byDay = new Map();
    for (let r = 1; r < lines.length; r++) {
      const cells = lines[r].split(/[,;\t]/);
      const ts = (cells[ti] || '').replace(/"/g, '').trim(), v = parseFloat(cells[ai]);
      const m = ts.match(/^(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})/);
      if (!m || !isFinite(v)) continue;
      const minute = +m[2] * 60 + +m[3];
      if (!byDay.has(m[1])) byDay.set(m[1], new Map());
      byDay.get(m[1]).set(minute, Math.max(0, v));
    }
    const days = [];
    for (const [date, mins] of [...byDay.entries()].sort()) {
      if (mins.size !== 1440) continue;
      days.push({ date, minutes: Array.from({ length: 1440 }, (_, i) => mins.get(i)) });
    }
    return days;
  }
  return { predictDay, binDay, parseCSV };
})();
if (typeof module !== 'undefined') module.exports = CNN;
