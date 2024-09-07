import { EPS } from "./api.js";
const mod = (a, b) => a - b * Math.floor(a / b);
const fract = (x) => x - Math.floor(x);
const trunc = (x) => x < 0 ? Math.ceil(x) : Math.floor(x);
const roundTo = (x, prec = 1) => Math.round(x / prec) * prec;
const floorTo = (x, prec = 1) => Math.floor(x / prec) * prec;
const ceilTo = (x, prec = 1) => Math.ceil(x / prec) * prec;
const roundEps = (x, eps = EPS) => {
  const f = fract(x);
  return f <= eps || f >= 1 - eps ? Math.round(x) : x;
};
export {
  ceilTo,
  floorTo,
  fract,
  mod,
  roundEps,
  roundTo,
  trunc
};
