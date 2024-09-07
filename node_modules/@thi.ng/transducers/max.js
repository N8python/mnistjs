import { reduce, reducer } from "./reduce.js";
function max(src) {
  return src ? reduce(max(), src) : reducer(
    () => -Infinity,
    (acc, x) => Math.max(acc, x)
  );
}
export {
  max
};
