import { reduce, reducer } from "./reduce.js";
function maxMag(src) {
  return src ? reduce(maxMag(), src) : reducer(
    () => 0,
    (acc, x) => Math.abs(x) > Math.abs(acc) ? x : acc
  );
}
export {
  maxMag
};
