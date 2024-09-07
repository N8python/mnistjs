const copysign = (x, y) => Math.sign(y) * Math.abs(x);
const exp2 = (x) => 2 ** x;
const fdim = (x, y) => Math.max(x - y, 0);
const fma = (x, y, z) => x * y + z;
const fmod = (x, y) => x % y;
const frexp = (x) => {
  if (x === 0 || !isFinite(x)) return [x, 0];
  const abs = Math.abs(x);
  let exp = Math.max(-1023, Math.floor(Math.log2(abs)) + 1);
  let y = abs * 2 ** -exp;
  while (y < 0.5) {
    y *= 2;
    exp--;
  }
  while (y >= 1) {
    y *= 0.5;
    exp++;
  }
  return [x < 0 ? -y : y, exp];
};
const ldexp = (x, exp) => x * 2 ** exp;
const remainder = (x, y) => x - y * Math.round(x / y);
const ldiv = (x, y) => {
  x |= 0;
  y |= 0;
  const q = x / y | 0;
  return [q, x - q * y];
};
export {
  copysign,
  exp2,
  fdim,
  fma,
  fmod,
  frexp,
  ldexp,
  ldiv,
  remainder
};
