import { reduce, reducer } from "./reduce.js";
function min(src) {
  return src ? reduce(min(), src) : reducer(
    () => Infinity,
    (acc, x) => Math.min(acc, x)
  );
}
export {
  min
};
