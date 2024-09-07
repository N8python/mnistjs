import { EPS } from "./api.js";
const abs = Math.abs;
const max = Math.max;
const eqDelta = (a, b, eps = EPS) => abs(a - b) <= eps;
const eqDeltaScaled = (a, b, eps = EPS) => abs(a - b) <= eps * max(1, abs(a), abs(b));
export {
  eqDelta,
  eqDeltaScaled
};
