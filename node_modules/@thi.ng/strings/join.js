import { memoizeO } from "@thi.ng/memoize/memoizeo";
const join = memoizeO(
  (sep) => (x) => x.join(sep)
);
export {
  join
};
