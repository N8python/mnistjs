const __defBits = (n) => new Array(n).fill(0).map((_, i) => 1 << n - 1 - i);
const MSB_BITS8 = __defBits(8);
const MSB_BITS16 = __defBits(16);
const MSB_BITS32 = __defBits(32);
const MASKS = new Array(33).fill(0).map((_, i) => 2 ** i - 1);
export {
  MASKS,
  MSB_BITS16,
  MSB_BITS32,
  MSB_BITS8
};
