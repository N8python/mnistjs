import { $$reduce, reducer } from "./reduce.js";
function count(...args) {
  const res = $$reduce(count, args);
  if (res !== void 0) {
    return res;
  }
  const [offset = 0, step = 1] = args;
  return reducer(
    () => offset,
    (acc, _) => acc + step
  );
}
export {
  count
};
