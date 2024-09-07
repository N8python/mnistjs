import { DEG2RAD, HALF_PI, INV_HALF_PI, PI, RAD2DEG, TAU } from "./api.js";
const sincos = (theta, n = 1) => [
  Math.sin(theta) * n,
  Math.cos(theta) * n
];
const cossin = (theta, n = 1) => [
  Math.cos(theta) * n,
  Math.sin(theta) * n
];
const absTheta = (theta) => (theta %= TAU, theta < 0 ? TAU + theta : theta);
const absInnerAngle = (theta) => (theta = Math.abs(theta), theta > PI ? TAU - theta : theta);
const angleDist = (a, b) => absInnerAngle(absTheta(b % TAU - a % TAU));
const atan2Abs = (y, x) => absTheta(Math.atan2(y, x));
const quadrant = (theta) => absTheta(theta) * INV_HALF_PI | 0;
const deg = (theta) => theta * RAD2DEG;
const rad = (theta) => theta * DEG2RAD;
const csc = (theta) => 1 / Math.sin(theta);
const sec = (theta) => 1 / Math.cos(theta);
const cot = (theta) => 1 / Math.tan(theta);
const loc = (a, b, gamma) => Math.sqrt(a * a + b * b - 2 * a * b * Math.cos(gamma));
const normCos = (x) => {
  const x2 = x * x;
  return 1 + x2 * (-4 + 2 * x2);
};
const __fastCos = (x) => {
  const x2 = x * x;
  return 0.99940307 + x2 * (-0.49558072 + 0.03679168 * x2);
};
const fastCos = (theta) => {
  theta %= TAU;
  theta < 0 && (theta = -theta);
  switch (theta * INV_HALF_PI | 0) {
    case 0:
      return __fastCos(theta);
    case 1:
      return -__fastCos(PI - theta);
    case 2:
      return -__fastCos(theta - PI);
    default:
      return __fastCos(TAU - theta);
  }
};
const fastSin = (theta) => fastCos(HALF_PI - theta);
const fromDMS = (deg2, min, sec2) => deg2 + min / 60 + sec2 / 3600;
const toDMS = (theta) => {
  const sign = Math.sign(theta);
  theta = Math.abs(theta);
  const deg2 = Math.trunc(theta);
  theta = (theta - deg2) * 60;
  const min = Math.trunc(theta);
  return [deg2 * sign, min, (theta - min) * 60];
};
export {
  absInnerAngle,
  absTheta,
  angleDist,
  atan2Abs,
  cossin,
  cot,
  csc,
  deg,
  fastCos,
  fastSin,
  fromDMS,
  loc,
  normCos,
  quadrant,
  rad,
  sec,
  sincos,
  toDMS
};
