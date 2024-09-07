import { assert } from "@thi.ng/errors/assert";
import { SYSTEM } from "./system.js";
const uniqueValuesFrom = (k, fn, existing = [], maxTrials = 100) => {
  let n = 0;
  while (n < k) {
    let i;
    let trials = maxTrials;
    do {
      i = fn();
    } while (existing.includes(i) && --trials > 0);
    if (trials <= 0) break;
    existing.push(i);
    n++;
  }
  return existing;
};
const uniqueIndices = (k, max, existing, maxTrials = max, rnd = SYSTEM) => {
  assert(k >= 0 && k <= max, `k must be in [0, ${max}] interval`);
  return uniqueValuesFrom(k, () => rnd.int() % max, existing, maxTrials);
};
const pickRandomUnique = (k, src, existing, maxTrials = 100, rnd = SYSTEM) => uniqueValuesFrom(k, () => src[rnd.int() % src.length], existing, maxTrials);
export {
  pickRandomUnique,
  uniqueIndices,
  uniqueValuesFrom
};
