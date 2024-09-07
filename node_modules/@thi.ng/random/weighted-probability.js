import { SYSTEM } from "./system.js";
const weightedProbability = (weight, rnd = SYSTEM) => {
  const a = rnd.float();
  const b = rnd.float();
  return [a, b < weight(a)];
};
export {
  weightedProbability
};
