import { reduce, reducer } from "./reduce.js";
function assocMap(src) {
  return src ? reduce(assocMap(), src) : reducer(
    () => /* @__PURE__ */ new Map(),
    (acc, [k, v]) => acc.set(k, v)
  );
}
export {
  assocMap
};
