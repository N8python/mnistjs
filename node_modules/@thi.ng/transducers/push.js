import { reducer } from "./reduce.js";
function push(src) {
  return src ? [...src] : reducer(
    () => [],
    (acc, x) => (acc.push(x), acc)
  );
}
export {
  push
};
