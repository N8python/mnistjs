import { isReduced } from "../reduced.js";
const __drain = (buf, complete, reduce) => (acc) => {
  while (buf.length && !isReduced(acc)) {
    acc = reduce(acc, buf.shift());
  }
  return complete(acc);
};
export {
  __drain
};
