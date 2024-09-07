import { reduce, reducer } from "./reduce.js";
function assocObj(src) {
  return src ? reduce(assocObj(), src) : reducer(
    () => ({}),
    (acc, [k, v]) => (acc[k] = v, acc)
  );
}
export {
  assocObj
};
