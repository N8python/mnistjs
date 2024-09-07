const clamp = (x, min, max) => x < min ? min : x > max ? max : x;
const clamp0 = (x) => x > 0 ? x : 0;
const clamp01 = (x) => x < 0 ? 0 : x > 1 ? 1 : x;
const clamp11 = (x) => x < -1 ? -1 : x > 1 ? 1 : x;
const clamp05 = (x) => x < 0 ? 0 : x > 0.5 ? 0.5 : x;
const minMax = (x, y) => x < y ? [x, y] : [y, x];
const wrap = (x, min, max) => {
  if (min === max) return min;
  if (x > max) {
    const d = max - min;
    x -= d;
    if (x > max) x -= d * ((x - min) / d | 0);
  } else if (x < min) {
    const d = max - min;
    x += d;
    if (x < min) x += d * ((min - x) / d + 1 | 0);
  }
  return x;
};
const wrapOnce = (x, min, max) => x < min ? x - min + max : x > max ? x - max + min : x;
const wrap01 = (x) => x < 0 ? x + 1 : x > 1 ? x - 1 : x;
const wrap11 = (x) => x < -1 ? x + 2 : x > 1 ? x - 2 : x;
const min2id = (a, b) => a <= b ? 0 : 1;
const min3id = (a, b, c) => a <= b ? a <= c ? 0 : 2 : b <= c ? 1 : 2;
const min4id = (a, b, c, d) => a <= b ? a <= c ? a <= d ? 0 : 3 : c <= d ? 2 : 3 : b <= c ? b <= d ? 1 : 3 : c <= d ? 2 : 3;
const max2id = (a, b) => a >= b ? 0 : 1;
const max3id = (a, b, c) => a >= b ? a >= c ? 0 : 2 : b >= c ? 1 : 2;
const max4id = (a, b, c, d) => a >= b ? a >= c ? a >= d ? 0 : 3 : c >= d ? 2 : 3 : b >= c ? b >= d ? 1 : 3 : c >= d ? 2 : 3;
const minNonZero2 = (a, b) => a !== 0 ? b !== 0 ? Math.min(a, b) : a : b;
const minNonZero3 = (a, b, c) => minNonZero2(minNonZero2(a, b), c);
const smin = (a, b, k) => smax(a, b, -k);
const smax = (a, b, k) => {
  const ea = Math.exp(a * k);
  const eb = Math.exp(b * k);
  return (a * ea + b * eb) / (ea + eb);
};
const sclamp = (x, min, max, k) => smin(smax(x, min, k), max, k);
const absMin = (a, b) => Math.abs(a) < Math.abs(b) ? a : b;
const absMax = (a, b) => Math.abs(a) > Math.abs(b) ? a : b;
const foldback = (e, x) => x < -e || x > e ? Math.abs(Math.abs((x - e) % (4 * e)) - 2 * e) - e : x;
const foldback01 = (x) => (x = Math.abs(x) % 2) > 1 ? 2 - x : x;
const inRange = (x, min, max) => x >= min && x <= max;
const inOpenRange = (x, min, max) => x > min && x < max;
export {
  absMax,
  absMin,
  clamp,
  clamp0,
  clamp01,
  clamp05,
  clamp11,
  foldback,
  foldback01,
  inOpenRange,
  inRange,
  max2id,
  max3id,
  max4id,
  min2id,
  min3id,
  min4id,
  minMax,
  minNonZero2,
  minNonZero3,
  sclamp,
  smax,
  smin,
  wrap,
  wrap01,
  wrap11,
  wrapOnce
};
