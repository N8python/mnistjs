const popCount = (x) => (x = x - (x >>> 1 & 1431655765), x = (x & 858993459) + (x >>> 2 & 858993459), (x + (x >>> 4) & 252645135) * 16843009 >>> 24);
const popCountArray = (data, start = 0, n = data.length) => {
  let num = 0;
  for (let end = start + n; start < end; start++) {
    const x = data[start];
    x > 0 && (num += popCount(x));
  }
  return num;
};
const hammingDist = (x, y) => popCount(x ^ y);
const clz32 = (x) => x !== 0 ? 31 - (Math.log(x >>> 0) / Math.LN2 | 0) : 32;
const ctz32 = (x) => {
  let c = 32;
  x &= -x;
  x && c--;
  x & 65535 && (c -= 16);
  x & 16711935 && (c -= 8);
  x & 252645135 && (c -= 4);
  x & 858993459 && (c -= 2);
  x & 1431655765 && (c -= 1);
  return c;
};
const bitSize = (x) => x > 1 ? Math.ceil(Math.log2(x)) : 0;
export {
  bitSize,
  clz32,
  ctz32,
  hammingDist,
  popCount,
  popCountArray
};
