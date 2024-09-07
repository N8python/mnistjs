const __eqStrict = (a, b) => a === b;
const levenshtein = (a, b, maxDist = Infinity, equiv = __eqStrict) => {
  if (a === b) {
    return 0;
  }
  if (a.length > b.length) {
    const tmp = a;
    a = b;
    b = tmp;
  }
  let la = a.length;
  let lb = b.length;
  while (la > 0 && equiv(a[~-la], b[~-lb])) {
    la--;
    lb--;
  }
  let offset = 0;
  while (offset < la && equiv(a[offset], b[offset])) {
    offset++;
  }
  la -= offset;
  lb -= offset;
  if (la === 0 || lb < 3) {
    return lb;
  }
  let x = 0;
  let y;
  let minDist;
  let d0;
  let d1;
  let d2;
  let d3;
  let dd;
  let dy;
  let ay;
  let bx0;
  let bx1;
  let bx2;
  let bx3;
  const _min = (d02, d12, d22, bx, ay2) => {
    return d02 < d12 || d22 < d12 ? d02 > d22 ? d22 + 1 : d02 + 1 : equiv(ay2, bx) ? d12 : d12 + 1;
  };
  const vector = [];
  for (y = 0; y < la; y++) {
    vector.push(y + 1, a[offset + y]);
  }
  const len = vector.length - 1;
  const lb3 = lb - 3;
  for (; x < lb3; ) {
    bx0 = b[offset + (d0 = x)];
    bx1 = b[offset + (d1 = x + 1)];
    bx2 = b[offset + (d2 = x + 2)];
    bx3 = b[offset + (d3 = x + 3)];
    dd = x += 4;
    minDist = Infinity;
    for (y = 0; y < len; y += 2) {
      dy = vector[y];
      ay = vector[y + 1];
      d0 = _min(dy, d0, d1, bx0, ay);
      d1 = _min(d0, d1, d2, bx1, ay);
      d2 = _min(d1, d2, d3, bx2, ay);
      dd = _min(d2, d3, dd, bx3, ay);
      dd < minDist && (minDist = dd);
      vector[y] = dd;
      d3 = d2;
      d2 = d1;
      d1 = d0;
      d0 = dy;
    }
    if (minDist > maxDist) return Infinity;
  }
  for (; x < lb; ) {
    bx0 = b[offset + (d0 = x)];
    dd = ++x;
    minDist = Infinity;
    for (y = 0; y < len; y += 2) {
      dy = vector[y];
      vector[y] = dd = _min(dy, d0, dd, bx0, vector[y + 1]);
      dd < minDist && (minDist = dd);
      d0 = dy;
    }
    if (minDist > maxDist) return Infinity;
  }
  return dd;
};
const normalizedLevenshtein = (a, b, maxDist = Infinity, equiv = __eqStrict) => {
  const n = Math.max(a.length, b.length);
  return n > 0 ? levenshtein(a, b, maxDist, equiv) / n : 0;
};
export {
  levenshtein,
  normalizedLevenshtein
};
