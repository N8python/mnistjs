import { reduce, reducer } from "./reduce.js";
function conj(src) {
  return src ? reduce(conj(), src) : reducer(
    () => /* @__PURE__ */ new Set(),
    (acc, x) => acc.add(x)
  );
}
export {
  conj
};
