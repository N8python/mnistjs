import { illegalArgs } from "@thi.ng/errors/illegal-arguments";
function partial(fn, ...args) {
  let [a, b, c, d, e, f, g, h] = args;
  switch (args.length) {
    case 1:
      return (...args2) => fn(a, ...args2);
    case 2:
      return (...args2) => fn(a, b, ...args2);
    case 3:
      return (...args2) => fn(a, b, c, ...args2);
    case 4:
      return (...args2) => fn(a, b, c, d, ...args2);
    case 5:
      return (...args2) => fn(a, b, c, d, e, ...args2);
    case 6:
      return (...args2) => fn(a, b, c, d, e, f, ...args2);
    case 7:
      return (...args2) => fn(a, b, c, d, e, f, g, ...args2);
    case 8:
      return (...args2) => fn(a, b, c, d, e, f, g, h, ...args2);
    default:
      illegalArgs();
  }
}
export {
  partial
};
