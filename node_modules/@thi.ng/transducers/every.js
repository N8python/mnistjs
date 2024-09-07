import { $$reduce, reducer } from "./reduce.js";
import { reduced } from "./reduced.js";
function every(...args) {
  const res = $$reduce(every, args);
  if (res !== void 0) return res;
  const pred = args[0];
  return reducer(
    () => true,
    pred ? (acc, x) => pred(x) ? acc : reduced(false) : (acc, x) => x ? acc : reduced(false)
  );
}
export {
  every
};
