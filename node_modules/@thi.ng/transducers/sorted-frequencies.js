import { identity } from "@thi.ng/compose/identity";
import { count } from "./count.js";
import { groupByMap } from "./group-by-map.js";
import { $$reduce } from "./reduce.js";
function sortedFrequencies(...args) {
  const res = $$reduce(sortedFrequencies, args);
  if (res) return res;
  const [init, complete, reduce] = groupByMap({
    key: args[0] || identity,
    group: count()
  });
  return [
    init,
    (acc) => [...complete(acc)].sort((a, b) => b[1] - a[1]),
    (acc, x) => reduce(acc, x)
  ];
}
export {
  sortedFrequencies
};
