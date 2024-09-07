const isPow2 = (x) => !!x && !(x & x - 1);
const ceilPow2 = (x) => {
  x += x === 0;
  --x;
  x |= x >>> 1;
  x |= x >>> 2;
  x |= x >>> 4;
  x |= x >>> 8;
  x |= x >>> 16;
  return x + 1;
};
const floorPow2 = (x) => {
  x |= x >>> 1;
  x |= x >>> 2;
  x |= x >>> 4;
  x |= x >>> 8;
  x |= x >>> 16;
  return x - (x >>> 1);
};
export {
  ceilPow2,
  floorPow2,
  isPow2
};
