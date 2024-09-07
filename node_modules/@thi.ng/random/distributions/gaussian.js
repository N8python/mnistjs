import { SYSTEM } from "../system.js";
const gaussian = (rnd = SYSTEM, n = 24, offset = 0, scale = 1) => () => {
  let sum = 0;
  let m = n;
  while (m-- > 0) sum += rnd.norm(scale);
  return sum / n + offset;
};
export {
  gaussian
};
