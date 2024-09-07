import { identity } from "@thi.ng/compose/identity";
import { groupByMap } from "./group-by-map.js";
import { normCount } from "./norm-count.js";
import { $$reduce } from "./reduce.js";
function normFrequencies(...args) {
  return $$reduce(normFrequencies, args) || groupByMap({
    key: args[1] || identity,
    group: normCount(args[0])
  });
}
export {
  normFrequencies
};
