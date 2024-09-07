import { illegalArgs } from "@thi.ng/errors/illegal-arguments";
const ensureIterable = (x) => {
  (x == null || !x[Symbol.iterator]) && illegalArgs(`value is not iterable: ${x}`);
  return x;
};
export {
  ensureIterable
};
