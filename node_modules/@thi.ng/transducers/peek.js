import { peek as _peek } from "@thi.ng/arrays/peek";
import { map } from "./map.js";
function peek(src) {
  return map(_peek, src);
}
export {
  peek
};
