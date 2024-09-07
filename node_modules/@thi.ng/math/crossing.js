import { EPS } from "./api.js";
import { eqDelta } from "./eqdelta.js";
const isCrossOver = (a1, a2, b1, b2) => a1 < b1 && a2 > b2;
const isCrossUnder = (a1, a2, b1, b2) => a1 > b1 && a2 < b2;
const classifyCrossing = (a1, a2, b1, b2, eps = EPS) => eqDelta(a1, b1, eps) && eqDelta(a2, b2, eps) ? eqDelta(a1, b2, eps) ? "flat" : "equal" : isCrossOver(a1, a2, b1, b2) ? "over" : isCrossUnder(a1, a2, b1, b2) ? "under" : "other";
export {
  classifyCrossing,
  isCrossOver,
  isCrossUnder
};
