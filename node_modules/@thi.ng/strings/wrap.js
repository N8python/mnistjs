import { memoizeJ } from "@thi.ng/memoize/memoizej";
const wrap = memoizeJ(
  (pad) => (x) => pad + x + pad
);
export {
  wrap
};
