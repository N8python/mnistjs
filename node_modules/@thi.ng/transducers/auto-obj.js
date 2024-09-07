import { reduce, reducer } from "./reduce.js";
function autoObj(prefix, src) {
  let id = 0;
  return src ? reduce(autoObj(prefix), src) : reducer(
    () => ({}),
    (acc, v) => (acc[prefix + id++] = v, acc)
  );
}
export {
  autoObj
};
