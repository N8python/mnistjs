import { isIterable } from "@thi.ng/checks/is-iterable";
import { iterator1 } from "./iterator.js";
import { map } from "./map.js";
function length(n = 0, src) {
  return isIterable(src) ? iterator1(length(n), src) : map(
    n === 0 ? (x) => x.length : (x) => x.length + n
  );
}
export {
  length
};
