import { __mathOp } from "./internal/mathop.js";
function add(...args) {
  return __mathOp(add, (acc, x) => acc + x, 0, args);
}
export {
  add
};
