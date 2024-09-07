import { reduce } from "./reduce.js";
import { isReduced, reduced } from "./reduced.js";
function reductions(rfn, src) {
  const [init, complete, _reduce] = rfn;
  return src ? reduce(reductions(rfn), src) : [
    () => [init()],
    (acc) => (acc[acc.length - 1] = complete(acc[acc.length - 1]), acc),
    (acc, x) => {
      const res = _reduce(acc[acc.length - 1], x);
      if (isReduced(res)) {
        acc.push(res.deref());
        return reduced(acc);
      }
      acc.push(res);
      return acc;
    }
  ];
}
export {
  reductions
};
