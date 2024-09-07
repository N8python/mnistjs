import { cat } from "./cat.js";
import { comp } from "./comp.js";
import { __iter, iterator } from "./iterator.js";
import { mapIndexed } from "./map-indexed.js";
function mapcatIndexed(...args) {
  return __iter(mapcatIndexed, args, iterator) || comp(mapIndexed(args[0], args[1]), cat());
}
export {
  mapcatIndexed
};
