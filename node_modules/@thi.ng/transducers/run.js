import { NO_OP } from "@thi.ng/api/api";
import { transduce } from "./transduce.js";
const NO_OP_REDUCER = [NO_OP, NO_OP, NO_OP];
function run(tx, ...args) {
  if (args.length === 1) {
    transduce(tx, NO_OP_REDUCER, args[0]);
  } else {
    const fx = args[0];
    transduce(tx, [NO_OP, NO_OP, (_, x) => fx(x)], args[1]);
  }
}
export {
  run
};
