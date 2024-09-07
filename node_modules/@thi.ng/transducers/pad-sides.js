import { concat } from "./concat.js";
import { repeat } from "./repeat.js";
const padSides = (src, x, numLeft = 1, numRight = numLeft) => numLeft > 0 ? numRight > 0 ? concat(repeat(x, numLeft), src, repeat(x, numRight)) : concat(repeat(x, numLeft), src) : numRight > 0 ? concat(src, repeat(x, numRight)) : concat(src);
export {
  padSides
};
