import { memoizeJ } from "@thi.ng/memoize/memoizej";
import { float } from "./float.js";
const vector = memoizeJ(
  (size, prec = 3, d = ",", pre = "[", post = "]") => {
    const f = typeof prec === "number" ? float(prec) : prec;
    switch (size) {
      case 1:
        return (v) => `${pre}${f(v[0])}${post}`;
      case 2:
        return (v) => `${pre}${f(v[0])}${d}${f(v[1])}${post}`;
      case 3:
        return (v) => `${pre}${f(v[0])}${d}${f(v[1])}${d}${f(v[2])}${post}`;
      case 4:
        return (v) => `${pre}${f(v[0])}${d}${f(v[1])}${d}${f(v[2])}${d}${f(
          v[3]
        )}${post}`;
      default:
        return (v) => {
          const res = [];
          for (let i = 0; i < v.length; i++) {
            res.push(f(v[i]));
          }
          return `${pre}${res.join(d)}${post}`;
        };
    }
  }
);
export {
  vector
};
