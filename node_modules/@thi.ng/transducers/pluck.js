import { isIterable } from "@thi.ng/checks/is-iterable";
import { iterator1 } from "./iterator.js";
import { map } from "./map.js";
function pluck(key, src) {
  return isIterable(src) ? iterator1(pluck(key), src) : map((x) => x[key]);
}
export {
  pluck
};
