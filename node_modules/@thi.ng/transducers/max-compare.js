import { __compareOp } from "./internal/mathop.js";
function maxCompare(...args) {
  return __compareOp(maxCompare, args, 1);
}
export {
  maxCompare
};
