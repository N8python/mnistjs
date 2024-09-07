import { SYSTEM } from "../system.js";
const geometric = (rnd = SYSTEM, p = 0.5) => p <= 0 ? () => Infinity : p >= 1 ? () => 1 : (p = Math.log(1 - p), () => Math.floor(Math.log(1 - rnd.float(1)) / p) + 1);
export {
  geometric
};
