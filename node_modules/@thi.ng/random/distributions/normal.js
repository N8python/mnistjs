import { SYSTEM } from "../system.js";
const normal = (rnd = SYSTEM, bias = 0, sigma = 1) => {
  let a;
  let b;
  let r;
  return () => {
    if (a != null) {
      b = a;
      a = null;
    } else {
      do {
        a = rnd.norm();
        b = rnd.norm();
        r = a * a + b * b;
      } while (r > 1 || r === 0);
    }
    return bias + sigma * b * Math.sqrt(-2 * Math.log(r) / r);
  };
};
export {
  normal
};
