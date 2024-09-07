import { __mathOp } from "./internal/mathop.js";
function sub(...args) {
  return __mathOp(sub, (acc, x) => acc - x, 0, args);
}
export {
  sub
};
