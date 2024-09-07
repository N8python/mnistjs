import { reduce, reducer } from "./reduce.js";
function div(init, src) {
  return src ? reduce(div(init), src) : reducer(
    () => init,
    (acc, x) => acc / x
  );
}
export {
  div
};
