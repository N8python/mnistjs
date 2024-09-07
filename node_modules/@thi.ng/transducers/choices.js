import { ensureArray } from "@thi.ng/arrays/ensure-array";
import { SYSTEM } from "@thi.ng/random/system";
import { weightedRandom } from "@thi.ng/random/weighted-random";
import { repeatedly } from "./repeatedly.js";
const choices = (choices2, weights, rnd = SYSTEM) => repeatedly(
  weights ? weightedRandom(ensureArray(choices2), weights, rnd) : () => choices2[rnd.float(choices2.length) | 0]
);
export {
  choices
};
