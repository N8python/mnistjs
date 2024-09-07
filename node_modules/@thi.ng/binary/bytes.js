import { floatToUintBits, floatToUintBits64 } from "./float.js";
const bytes16 = (x, le = false) => {
  const b0 = x & 255;
  const b1 = x >> 8 & 255;
  return le ? [b0, b1] : [b1, b0];
};
const bytes24 = (x, le = false) => {
  const b0 = x & 255;
  const b1 = x >> 8 & 255;
  const b2 = x >> 16 & 255;
  return le ? [b0, b1, b2] : [b2, b1, b0];
};
const bytes32 = (x, le = false) => {
  const b0 = x & 255;
  const b1 = x >> 8 & 255;
  const b2 = x >> 16 & 255;
  const b3 = x >> 24 & 255;
  return le ? [b0, b1, b2, b3] : [b3, b2, b1, b0];
};
const bytes64 = (hi, lo, le = false) => {
  return le ? bytes32(lo, le).concat(bytes32(hi, le)) : bytes32(hi, le).concat(bytes32(lo, le));
};
const bytesF32 = (x, le = false) => bytes32(floatToUintBits(x), le);
const bytesF64 = (x, le = false) => (
  //@ts-ignore
  bytes64(...floatToUintBits64(x), le)
);
export {
  bytes16,
  bytes24,
  bytes32,
  bytes64,
  bytesF32,
  bytesF64
};
