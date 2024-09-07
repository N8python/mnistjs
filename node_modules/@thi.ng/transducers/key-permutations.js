import { assocObj } from "./assoc-obj.js";
import { map } from "./map.js";
import { mapcat } from "./mapcat.js";
import { pairs } from "./pairs.js";
import { partition } from "./partition.js";
import { permutations } from "./permutations.js";
const keyPermutations = (spec) => map(
  (x) => assocObj(partition(2, x)),
  permutations(...mapcat(([k, v]) => [[k], v], pairs(spec)))
);
export {
  keyPermutations
};
