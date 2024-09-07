import { memoizeJ } from "@thi.ng/memoize/memoizej";
const truncate = memoizeJ(
  (n, suffix = "") => (x) => x.length > n ? x.substring(0, n - suffix.length) + suffix : x
);
const truncateRight = truncate;
export {
  truncate,
  truncateRight
};
