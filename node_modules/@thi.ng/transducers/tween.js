import { normRange } from "./norm-range.js";
import { repeat } from "./repeat.js";
function* tween(opts) {
  const { min, max, num, init, mix, stops } = opts;
  const easing = opts.easing || ((x) => x);
  let l = stops.length;
  if (l < 1) return;
  if (l === 1) {
    yield* repeat(mix(init(stops[0][1], stops[0][1]), 0), num);
  }
  stops.sort((a, b) => a[0] - b[0]);
  stops[l - 1][0] < max && stops.push([max, stops[l - 1][1]]);
  stops[0][0] > min && stops.unshift([min, stops[0][1]]);
  const range = max - min;
  let start = stops[0][0];
  let end = stops[1][0];
  let delta = end - start;
  let interval = init(stops[0][1], stops[1][1]);
  let i = 1;
  l = stops.length;
  for (let t of normRange(num)) {
    t = min + range * t;
    if (t > end) {
      while (i < l && t > stops[i][0]) i++;
      start = stops[i - 1][0];
      end = stops[i][0];
      delta = end - start;
      interval = init(stops[i - 1][1], stops[i][1]);
    }
    yield mix(interval, easing(delta !== 0 ? (t - start) / delta : 0));
  }
}
export {
  tween
};
