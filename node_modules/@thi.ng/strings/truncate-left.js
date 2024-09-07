import { memoizeJ } from "@thi.ng/memoize/memoizej";
const truncateLeft = memoizeJ(
  (n, prefix = "") => (x) => x.length > n ? prefix + x.substring(x.length - n + prefix.length) : x
);
export {
  truncateLeft
};
