const lane16 = (x, lane) => x >>> (1 - lane << 4) & 65535;
const lane8 = (x, lane) => x >>> (3 - lane << 3) & 255;
const lane4 = (x, lane) => x >>> (7 - lane << 2) & 15;
const lane2 = (x, lane) => x >>> (15 - lane << 1) & 3;
const setLane16 = (x, y, lane) => lane ? mux(x, y, 65535) : mux(x, y << 16, 4294901760);
const setLane8 = (x, y, lane) => {
  const l = 3 - lane << 3;
  return (~(255 << l) & x | (y & 255) << l) >>> 0;
};
const setLane4 = (x, y, lane) => {
  const l = 7 - lane << 2;
  return (~(15 << l) & x | (y & 15) << l) >>> 0;
};
const setLane2 = (x, y, lane) => {
  const l = 15 - lane << 1;
  return (~(3 << l) & x | (y & 3) << l) >>> 0;
};
const swizzle8 = (x, a, b, c, d) => (lane8(x, a) << 24 | lane8(x, b) << 16 | lane8(x, c) << 8 | lane8(x, d)) >>> 0;
const swizzle4 = (x, a, b, c, d, e, f, g, h) => (lane4(x, a) << 28 | lane4(x, b) << 24 | lane4(x, c) << 20 | lane4(x, d) << 16 | lane4(x, e) << 12 | lane4(x, f) << 8 | lane4(x, g) << 4 | lane4(x, h)) >>> 0;
const mux = (a, b, mask) => ~mask & a | mask & b;
const flip8 = (x) => (x >>> 24 | x >> 8 & 65280 | (x & 65280) << 8 | x << 24) >>> 0;
const flip16 = (x) => mux(x << 16, x >>> 16, 65535);
const flipBytes = flip8;
const swapLane02 = (x) => (x & 65280) << 16 | x >>> 16 & 65280 | x & 16711935;
const swapLane13 = (x) => (x & 255) << 16 | x >> 16 & 255 | x & 4278255360;
export {
  flip16,
  flip8,
  flipBytes,
  lane16,
  lane2,
  lane4,
  lane8,
  mux,
  setLane16,
  setLane2,
  setLane4,
  setLane8,
  swapLane02,
  swapLane13,
  swizzle4,
  swizzle8
};
