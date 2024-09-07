import { clz32 } from "./count.js";
const binaryOneHot = (x) => 1 << x >>> 0;
const oneHotBinary = (x) => 31 - clz32(x);
export {
  binaryOneHot,
  oneHotBinary
};
