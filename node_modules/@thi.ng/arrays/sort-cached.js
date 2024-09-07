import { isFunction } from "@thi.ng/checks/is-function";
import { compare } from "@thi.ng/compare/compare";
import { assert } from "@thi.ng/errors/assert";
import { quickSort } from "./quicksort.js";
import { multiSwap } from "./swap.js";
const sortByCachedKey = (src, key, cmp = compare) => {
  const keys = isFunction(key) ? src.map(key) : key;
  assert(keys.length === src.length, `keys.length != src.length`);
  quickSort(keys, cmp, multiSwap(src));
  return src;
};
export {
  sortByCachedKey
};
