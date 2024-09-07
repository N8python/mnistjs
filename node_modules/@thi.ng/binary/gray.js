const encodeGray32 = (x) => (x ^ x >>> 1) >>> 0;
const decodeGray32 = (x) => {
  x = x ^ x >>> 16;
  x = x ^ x >>> 8;
  x = x ^ x >>> 4;
  x = x ^ x >>> 2;
  x = x ^ x >>> 1;
  return x >>> 0;
};
export {
  decodeGray32,
  encodeGray32
};
