import { groupByObj } from "./group-by-obj.js";
import { push } from "./push.js";
const __branchPred = (key, b, l, r) => (x) => key(x) & b ? r : l;
const groupBinary = (bits, key, branch, leaf, left = "l", right = "r") => {
  const init = branch || (() => ({}));
  let rfn = groupByObj({
    key: __branchPred(key, 1, left, right),
    group: leaf || push()
  });
  for (let i = 2, maxIndex = 1 << bits; i < maxIndex; i <<= 1) {
    rfn = groupByObj({
      key: __branchPred(key, i, left, right),
      group: [init, rfn[1], rfn[2]]
    });
  }
  return [init, rfn[1], rfn[2]];
};
export {
  groupBinary
};
