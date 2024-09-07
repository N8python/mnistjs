import { SYSTEM } from "./system.js";
const randomID = (len = 4, prefix = "", syms = "abcdefghijklmnopqrstuvwxyz", rnd = SYSTEM) => {
  const n = syms.length;
  for (; len-- > 0; ) {
    prefix += syms[rnd.int() % n];
  }
  return prefix;
};
export {
  randomID
};
