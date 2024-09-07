import { reducer } from "./reduce.js";
function pushKeys(key, src) {
  return src ? [...src].map((x) => x[key]) : reducer(
    () => [],
    (acc, x) => (acc.push(x[key]), acc)
  );
}
export {
  pushKeys
};
