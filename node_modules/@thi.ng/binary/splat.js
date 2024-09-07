const splat4_24 = (x) => (x & 15) * 1118481;
const splat4_32 = (x) => (x & 15) * 286331153 >>> 0;
const splat8_24 = (x) => (x & 255) * 65793;
const splat8_32 = (x) => (x & 255) * 16843009 >>> 0;
const splat16_32 = (x) => (x &= 65535, (x << 16 | x) >>> 0);
const same4 = (x) => (x >> 4 & 15) === (x & 15);
const same8 = (x) => (x >> 8 & 255) === (x & 255);
const interleave4_12_24 = (x) => (x & 3840) * 4352 | (x & 240) * 272 | (x & 15) * 17;
const interleave4_16_32 = (x) => ((x & 61440) * 69632 | (x & 3840) * 4352 | (x & 240) * 272 | (x & 15) * 17) >>> 0;
export {
  interleave4_12_24,
  interleave4_16_32,
  same4,
  same8,
  splat16_32,
  splat4_24,
  splat4_32,
  splat8_24,
  splat8_32
};
