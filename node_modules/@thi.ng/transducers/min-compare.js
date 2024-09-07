import { __compareOp } from "./internal/mathop.js";
function minCompare(...args) {
  return __compareOp(minCompare, args, -1);
}
export {
  minCompare
};
