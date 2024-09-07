import { reduce } from "./reduce.js";
function mean(src) {
  let n = 1;
  return src ? reduce(mean(), src) : [
    () => n = 0,
    (acc) => n > 1 ? acc / n : acc,
    (acc, x) => (n++, acc + x)
  ];
}
export {
  mean
};
