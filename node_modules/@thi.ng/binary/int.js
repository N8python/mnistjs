const i8u8 = (x) => x >>> 0 & 255;
const i16u16 = (x) => x >>> 0 & 65535;
const i32u32 = (x) => x >>> 0;
const u8i8 = (x) => (x &= 255, x - (x >> 7 << 8));
const u16i16 = (x) => (x &= 65535, x - (x >> 15 << 16));
const u24i24 = (x) => (x &= 16777215, x - (x >> 23 << 24));
const u32i32 = (x) => x | 0;
export {
  i16u16,
  i32u32,
  i8u8,
  u16i16,
  u24i24,
  u32i32,
  u8i8
};
