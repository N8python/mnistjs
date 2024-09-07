import { isNotStringAndIterable } from "@thi.ng/checks/is-not-string-iterable";
import { flattenWith } from "./flatten-with.js";
function flatten(src) {
  return flattenWith(
    (x) => isNotStringAndIterable(x) ? x : void 0,
    src
  );
}
export {
  flatten
};
