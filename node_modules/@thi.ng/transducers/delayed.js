import { delayed as _delayed } from "@thi.ng/compose/delayed";
import { map } from "./map.js";
const delayed = (t) => map((x) => _delayed(x, t));
export {
  delayed
};
