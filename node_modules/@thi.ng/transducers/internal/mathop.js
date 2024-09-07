import { compare } from "@thi.ng/compare/compare";
import { $$reduce, reducer } from "../reduce.js";
const __mathOp = (rfn, fn, initDefault, args) => {
  const res = $$reduce(rfn, args);
  if (res !== void 0) return res;
  const init = args[0] || initDefault;
  return reducer(() => init, fn);
};
const __compareOp = (rfn, args, sign) => {
  const res = $$reduce(rfn, args);
  if (res !== void 0) return res;
  const init = args[0];
  const cmp = args[1] || compare;
  return reducer(init, (acc, x) => sign * cmp(acc, x) >= 0 ? acc : x);
};
export {
  __compareOp,
  __mathOp
};
