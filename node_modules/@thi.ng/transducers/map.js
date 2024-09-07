import { isIterable } from "@thi.ng/checks/is-iterable";
import { compR } from "./compr.js";
import { iterator1 } from "./iterator.js";
function map(fn, src) {
  return isIterable(src) ? iterator1(map(fn), src) : (rfn) => {
    const r = rfn[2];
    return compR(rfn, (acc, x) => r(acc, fn(x)));
  };
}
const mapA = (fn, src) => [
  ...map(fn, src)
];
export {
  map,
  mapA
};
