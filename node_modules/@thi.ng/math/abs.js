import { EPS } from "./api.js";
const absDiff = (a, b) => Math.abs(a - b);
const sign = (x, eps = EPS) => x > eps ? 1 : x < -eps ? -1 : 0;
const signedPow = (x, k, eps = EPS) => sign(x, eps) * Math.abs(x) ** k;
export {
  absDiff,
  sign,
  signedPow
};
