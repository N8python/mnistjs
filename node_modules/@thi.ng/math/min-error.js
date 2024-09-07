import { EPS } from "./api.js";
const minError = (fn, error, q, res = 16, iter = 8, start = 0, end = 1, eps = EPS) => {
  if (iter <= 0) return (start + end) / 2;
  const delta = (end - start) / res;
  let minT = start;
  let minE = Infinity;
  for (let i = 0; i <= res; i++) {
    const t = start + i * delta;
    const e = error(q, fn(t));
    if (e < minE) {
      if (e <= eps) return t;
      minE = e;
      minT = t;
    }
  }
  return minError(
    fn,
    error,
    q,
    res,
    iter - 1,
    Math.max(minT - delta, 0),
    Math.min(minT + delta, 1)
  );
};
export {
  minError
};
