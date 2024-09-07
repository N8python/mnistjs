import { compare } from "@thi.ng/compare/compare";
import { fillRange } from "./fill-range.js";
import { sortByCachedKey } from "./sort-cached.js";
const argSort = (src, cmp = compare) => {
  const $src = [...src];
  return sortByCachedKey(
    fillRange(new Array($src.length)),
    $src,
    cmp
  );
};
export {
  argSort
};
