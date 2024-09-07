import { $$reduce } from "./reduce.js";
function normCount(...args) {
  const res = $$reduce(normCount, args);
  if (res !== void 0) return res;
  const norm = args[0];
  return [() => 0, (acc) => acc / norm, (acc) => acc + 1];
}
export {
  normCount
};
