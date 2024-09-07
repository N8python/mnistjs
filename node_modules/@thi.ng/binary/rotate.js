const rotateLeft = (x, n) => (x << n | x >>> 32 - n) >>> 0;
const rotateRight = (x, n) => (x >>> n | x << 32 - n) >>> 0;
const shiftRL = (x, n) => n < 0 ? x << -n : x >>> n;
export {
  rotateLeft,
  rotateRight,
  shiftRL
};
