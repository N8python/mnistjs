import { defMask } from "./mask.js";
const bitClear = (x, bit) => (x & ~(1 << bit)) >>> 0;
const bitFlip = (x, bit) => (x ^ 1 << bit) >>> 0;
const bitSet = (x, bit) => (x | 1 << bit) >>> 0;
const bitSetWindow = (x, y, from, to) => {
  const m = defMask(from, to);
  return x & ~m | y << (1 << from) & m;
};
const bitClearWindow = (x, from, to) => x & ~defMask(from, to);
export {
  bitClear,
  bitClearWindow,
  bitFlip,
  bitSet,
  bitSetWindow
};
