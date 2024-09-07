import { SYSTEM } from "../system.js";
const exponential = (rnd = SYSTEM, lambda = 10) => lambda === 0 ? () => Infinity : () => -Math.log(1 - rnd.float(1)) / lambda;
export {
  exponential
};
