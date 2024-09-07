import { isArray } from "@thi.ng/checks/is-array";
import { comp } from "./comp.js";
import { filter } from "./filter.js";
import { __iter } from "./iterator.js";
import { map } from "./map.js";
import { renamer } from "./renamer.js";
import { transduce } from "./transduce.js";
function rename(...args) {
  const iter = args.length > 2 && __iter(rename, args);
  if (iter) {
    return iter;
  }
  let [kmap, reducer] = args;
  if (isArray(kmap)) {
    kmap = kmap.reduce((acc, k, i) => (acc[k] = i, acc), {});
  }
  if (reducer) {
    const ks = Object.keys(kmap);
    return map(
      (y) => transduce(
        comp(
          map(
            (k) => [k, y[kmap[k]]]
          ),
          filter((x) => x[1] !== void 0)
        ),
        reducer,
        ks
      )
    );
  } else {
    return map(renamer(kmap));
  }
}
export {
  rename
};
