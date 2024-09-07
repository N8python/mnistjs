import { $$reduce, reducer } from "./reduce.js";
function fill(...args) {
  const res = $$reduce(fill, args);
  if (res !== void 0) return res;
  let start = args[0] || 0;
  return reducer(
    () => [],
    (acc, x) => (acc[start++] = x, acc)
  );
}
function fillN(...args) {
  return fill(...args);
}
export {
  fill,
  fillN
};
