import { SYSTEM } from "../system.js";
const uniform = (rnd = SYSTEM, min = 0, max = 1) => () => rnd.minmax(min, max);
export {
  uniform
};
