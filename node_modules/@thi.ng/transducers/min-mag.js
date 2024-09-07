import { reduce, reducer } from "./reduce.js";
function minMag(src) {
  return src ? reduce(minMag(), src) : reducer(
    () => Infinity,
    (acc, x) => Math.abs(x) < Math.abs(acc) ? x : acc
  );
}
export {
  minMag
};
