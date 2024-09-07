import { map } from "./map.js";
import { permutations } from "./permutations.js";
import { range, Range } from "./range.js";
import { zip } from "./zip.js";
const rangeNd = (min, max) => permutations.apply(
  null,
  max ? [...map(([a, b]) => range(a, b), zip(min, max))] : [...map(range, min)]
);
export {
  rangeNd
};
