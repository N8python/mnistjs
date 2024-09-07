import { clamp01, clamp11 } from "./interval.js";
const norm = (x, a, b) => b !== a ? (x - a) / (b - a) : 0;
const fit = (x, a, b, c, d) => c + (d - c) * norm(x, a, b);
const fitClamped = (x, a, b, c, d) => c + (d - c) * clamp01(norm(x, a, b));
const fit01 = (x, a, b) => a + (b - a) * clamp01(x);
const fit10 = (x, a, b) => b + (a - b) * clamp01(x);
const fit11 = (x, a, b) => a + (b - a) * (0.5 + 0.5 * clamp11(x));
export {
  fit,
  fit01,
  fit10,
  fit11,
  fitClamped,
  norm
};
