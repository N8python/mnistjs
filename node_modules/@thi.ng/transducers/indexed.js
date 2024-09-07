import { __iter } from "./iterator.js";
import { mapIndexed } from "./map-indexed.js";
function indexed(...args) {
  const iter = __iter(indexed, args);
  if (iter) {
    return iter;
  }
  const from = args[0] || 0;
  return mapIndexed((i, x) => [from + i, x]);
}
export {
  indexed
};
