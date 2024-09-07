import { map } from "./map.js";
import { normRange } from "./norm-range.js";
const line = (start, end, steps = 10) => {
  const delta = end - start;
  return map((t) => start + delta * t, normRange(steps));
};
export {
  line
};
