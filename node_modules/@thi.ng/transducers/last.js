import { NO_OP } from "@thi.ng/api/api";
import { reduce, reducer } from "./reduce.js";
function last(src) {
  return src ? reduce(last(), src) : reducer(NO_OP, (_, x) => x);
}
export {
  last
};
