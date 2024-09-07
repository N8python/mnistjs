import { HALF_PI, PI, TAU } from "./api.js";
const { cos, sin, sqrt } = Math;
const defEaseInExp = (k) => (t) => t ** k;
const defEaseOutExp = (k) => (t) => 1 - (1 - t) ** k;
const defEaseInOutExp = (k) => {
  const k2 = 2 ** (k - 1);
  return (t) => t < 0.5 ? k2 * t ** k : 1 - (-2 * t + 2) ** k / 2;
};
const easeLinear = (t) => t;
const easeIn2 = defEaseInExp(2);
const easeOut2 = defEaseOutExp(2);
const easeInOut2 = defEaseInOutExp(2);
const easeIn3 = defEaseInExp(3);
const easeOut3 = defEaseOutExp(3);
const easeInOut3 = defEaseInOutExp(3);
const easeIn4 = defEaseInExp(4);
const easeOut4 = defEaseOutExp(4);
const easeInOut4 = defEaseInOutExp(4);
const easeIn5 = defEaseInExp(5);
const easeOut5 = defEaseOutExp(5);
const easeInOut5 = defEaseInOutExp(5);
const easeInBack = (t) => 2.70158 * t ** 3 - 1.70158 * t ** 2;
const easeOutBack = (t) => 2.70158 * (t - 1) ** 3 + 1 + 1.70158 * (t - 1) ** 2;
const easeInOutBack = (t) => t < 0.5 ? (2 * t) ** 2 * ((2.5949095 + 1) * 2 * t - 2.5949095) / 2 : ((2 * t - 2) ** 2 * ((2.5949095 + 1) * (t * 2 - 2) + 2.5949095) + 2) / 2;
const easeInBounce = (t) => 1 - easeOutBounce(1 - t);
const easeOutBounce = (t) => t < 1 / 2.75 ? 7.5625 * (t * t) : t < 2 / 2.75 ? 7.5625 * (t - 1.5 / 2.75) * (t - 1.5 / 2.75) + 0.75 : t < 2.5 / 2.75 ? 7.5625 * (t - 2.25 / 2.75) * (t - 2.25 / 2.75) + 0.9375 : 7.5625 * (t - 2.625 / 2.75) * (t - 2.625 / 2.75) + 0.984375;
const easeInOutBounce = (t) => t < 0.5 ? (1 - easeOutBounce(1 - 2 * t)) / 2 : (1 + easeOutBounce(2 * t - 1)) / 2;
const easeInCirc = (t) => 1 - sqrt(1 - t ** 2);
const easeOutCirc = (t) => sqrt(1 - (t - 1) ** 2);
const easeInOutCirc = (t) => t < 0.5 ? (1 - sqrt(1 - (2 * t) ** 2)) / 2 : (sqrt(1 - (-2 * t + 2) ** 2) + 1) / 2;
const easeInElastic = (t) => t <= 0 ? 0 : t >= 1 ? 1 : -(2 ** (10 * t - 10)) * sin((t * 10 - 10.75) * (TAU / 3));
const easeOutElastic = (t) => t <= 0 ? 0 : t >= 1 ? 1 : 2 ** (-10 * t) * sin((t * 10 - 0.75) * (TAU / 3)) + 1;
const easeInOutElastic = (t) => t <= 0 ? 0 : t >= 1 ? 1 : t < 0.5 ? -(2 ** (20 * t - 10) * sin((20 * t - 11.125) * (TAU / 4.5))) / 2 : 2 ** (-20 * t + 10) * sin((20 * t - 11.125) * (TAU / 4.5)) / 2 + 1;
const easeInExp2 = (t) => t <= 0 ? 0 : 2 ** (10 * t - 10);
const easeOutExp2 = (t) => t >= 1 ? 1 : 1 - 2 ** (-10 * t);
const easeInOutExp2 = (t) => t < 0 ? 0 : t >= 1 ? 1 : t < 0.5 ? 2 ** (20 * t - 10) / 2 : (2 - 2 ** (-20 * t + 10)) / 2;
const easeInSine = (t) => 1 - cos(t * HALF_PI);
const easeOutSine = (t) => sin(t * HALF_PI);
const easeInOutSine = (t) => -(cos(PI * t) - 1) / 2;
export {
  defEaseInExp,
  defEaseInOutExp,
  defEaseOutExp,
  easeIn2,
  easeIn3,
  easeIn4,
  easeIn5,
  easeInBack,
  easeInBounce,
  easeInCirc,
  easeInElastic,
  easeInExp2,
  easeInOut2,
  easeInOut3,
  easeInOut4,
  easeInOut5,
  easeInOutBack,
  easeInOutBounce,
  easeInOutCirc,
  easeInOutElastic,
  easeInOutExp2,
  easeInOutSine,
  easeInSine,
  easeLinear,
  easeOut2,
  easeOut3,
  easeOut4,
  easeOut5,
  easeOutBack,
  easeOutBounce,
  easeOutCirc,
  easeOutElastic,
  easeOutExp2,
  easeOutSine
};
