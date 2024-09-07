import { clamp01 } from "./interval.js";
const step = (edge, x) => x < edge ? 0 : 1;
const smoothStep = (edge, edge2, x) => smoothStep01(clamp01((x - edge) / (edge2 - edge)));
const smoothStep01 = (x) => x * x * (3 - 2 * x);
const smootherStep = (edge, edge2, x) => smootherStep01(clamp01((x - edge) / (edge2 - edge)));
const smootherStep01 = (x) => x * x * x * (x * (x * 6 - 15) + 10);
const expStep = (k, n, x) => 1 - Math.exp(-k * Math.pow(x, n));
export {
  expStep,
  smoothStep,
  smoothStep01,
  smootherStep,
  smootherStep01,
  step
};
