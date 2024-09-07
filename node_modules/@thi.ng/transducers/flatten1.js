import { identity } from "@thi.ng/api/fn";
import { mapcat } from "./mapcat.js";
function flatten1(src) {
  return mapcat(identity, src);
}
export {
  flatten1
};
