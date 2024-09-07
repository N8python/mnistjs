import { assert } from "@thi.ng/errors/assert";
import { SYSTEM } from "./system.js";
const weightedRandom = (choices, weights, rnd = SYSTEM) => {
  const n = choices.length;
  assert(n > 0, "no choices given");
  const opts = weights ? choices.map((x, i) => [weights[i] || 0, x]).sort((a, b) => b[0] - a[0]) : choices.map((x) => [1, x]);
  const total = opts.reduce((acc, o) => acc + o[0], 0);
  total <= 0 && console.warn("total weights <= 0");
  return () => {
    const r = rnd.float(total);
    let sum = total;
    for (let i = 0; i < n; i++) {
      sum -= opts[i][0];
      if (sum <= r) {
        return opts[i][1];
      }
    }
    return void 0;
  };
};
const weightedRandomKey = (choices, rnd = SYSTEM) => {
  const keys = Object.keys(choices);
  return weightedRandom(
    keys,
    keys.map((x) => choices[x]),
    rnd
  );
};
export {
  weightedRandom,
  weightedRandomKey
};
