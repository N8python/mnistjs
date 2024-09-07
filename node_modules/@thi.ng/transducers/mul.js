import { __mathOp } from "./internal/mathop.js";
function mul(...args) {
  return __mathOp(mul, (acc, x) => acc * x, 1, args);
}
export {
  mul
};
