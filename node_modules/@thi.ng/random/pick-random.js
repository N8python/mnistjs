import { SYSTEM } from "./system.js";
const pickRandom = (src, rnd = SYSTEM, start = 0, end = src.length) => src[rnd.minmax(start, end) | 0];
const pickRandomKey = (obj, rnd = SYSTEM) => pickRandom(Object.keys(obj), rnd);
export {
  pickRandom,
  pickRandomKey
};
