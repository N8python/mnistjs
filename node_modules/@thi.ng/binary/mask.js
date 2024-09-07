import { MASKS } from "./constants.js";
const defMask = (a, b) => (~MASKS[a] & MASKS[b]) >>> 0;
const maskL = (n, x) => (x & MASKS[n]) >>> 0;
const maskH = (n, x) => (x & ~MASKS[n]) >>> 0;
export {
  defMask,
  maskH,
  maskL
};
