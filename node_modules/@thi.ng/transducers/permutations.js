import { ensureArrayLike } from "@thi.ng/arrays/ensure-array";
import { illegalArgs } from "@thi.ng/errors/illegal-arguments";
import { range } from "./range.js";
function* permutations(...src) {
  const n = src.length - 1;
  if (n < 0) {
    return;
  }
  const step = new Array(n + 1).fill(0);
  const realized = src.map(ensureArrayLike);
  const total = realized.reduce((acc, x) => acc * x.length, 1);
  for (let i = 0; i < total; i++) {
    const tuple = [];
    for (let j = n; j >= 0; j--) {
      const r = realized[j];
      let s = step[j];
      if (s === r.length) {
        step[j] = s = 0;
        j > 0 && step[j - 1]++;
      }
      tuple[j] = r[s];
    }
    step[n]++;
    yield tuple;
  }
}
const permutationsN = (n, m = n, offsets) => {
  if (offsets && offsets.length < n) {
    illegalArgs(`insufficient offsets, got ${offsets.length}, needed ${n}`);
  }
  const seqs = [];
  while (n-- > 0) {
    const o = offsets ? offsets[n] : 0;
    seqs[n] = range(o, o + m);
  }
  return permutations.apply(null, seqs);
};
export {
  permutations,
  permutationsN
};
