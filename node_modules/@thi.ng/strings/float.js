import { memoizeJ } from "@thi.ng/memoize/memoizej";
import { padLeft } from "./pad-left.js";
const float = memoizeJ(
  (prec, special = false) => special ? (x) => __nanOrInf(x) || x.toFixed(prec) : (x) => x.toFixed(prec)
);
const floatFixedWidth = memoizeJ((width, prec = 3) => {
  const l = width - prec - 1;
  const pl = Math.pow(10, l);
  const pln = -Math.pow(10, l - 1);
  const pr = Math.pow(10, -(prec - 1));
  const pad = padLeft(width);
  return (x) => {
    const ax = Math.abs(x);
    return pad(
      __nanOrInf(x) || (x === 0 ? "0" : ax < pr || ax >= pl ? __exp(x, width) : x.toFixed(prec - (x < pln ? 1 : 0)))
    );
  };
});
const __exp = (x, w) => x.toExponential(
  Math.max(
    w - 4 - (Math.log(Math.abs(x)) / Math.LN10 >= 10 ? 2 : 1) - (x < 0 ? 1 : 0),
    0
  )
);
const __nanOrInf = (x) => isNaN(x) ? "NaN" : x === Infinity ? "+\u221E" : x === -Infinity ? "-\u221E" : void 0;
export {
  float,
  floatFixedWidth
};
