import { __iter, iterator } from "./iterator.js";
import { partitionBy } from "./partition-by.js";
function wordWrap(...args) {
  const iter = __iter(wordWrap, args, iterator);
  if (iter) {
    return iter;
  }
  const lineLength = args[0];
  const { delim = 1, always = true } = args[1];
  return partitionBy(() => {
    let n = 0;
    let flag = false;
    return (w) => {
      n += w.length + delim;
      if (n > lineLength + (always ? 0 : delim)) {
        flag = !flag;
        n = w.length + delim;
      }
      return flag;
    };
  }, true);
}
export {
  wordWrap
};
