import { mixHermite } from "@thi.ng/math/mix";
import { interpolate } from "./interpolate.js";
function interpolateHermite(n, src) {
  return interpolate(
    (chunk, t) => mixHermite(...chunk, t),
    4,
    n,
    src
  );
}
export {
  interpolateHermite
};
