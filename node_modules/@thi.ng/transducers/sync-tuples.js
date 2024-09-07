import { __iter, iterator } from "./iterator.js";
function syncTuples(...args) {
  const iter = __iter(syncTuples, args, iterator);
  if (iter) return iter;
  const size = args[0];
  return ([init, complete, reduce]) => {
    const prev = new Array(size);
    return [
      init,
      complete,
      (acc, x) => {
        let partial = false;
        let filled = true;
        for (let i = 0; i < size; i++) {
          if (x[i] != null) {
            prev[i] = x[i];
            partial = true;
          } else if (prev[i] == null) filled = false;
        }
        return partial && filled ? reduce(acc, prev.slice()) : acc;
      }
    ];
  };
}
export {
  syncTuples
};
