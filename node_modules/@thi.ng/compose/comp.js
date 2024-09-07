import { illegalArity } from "@thi.ng/errors/illegal-arity";
function comp(...fns) {
  let [a, b, c, d, e, f, g, h, i, j] = fns;
  switch (fns.length) {
    case 0:
      illegalArity(0);
    case 1:
      return a;
    case 2:
      return (...args) => a(b(...args));
    case 3:
      return (...args) => a(b(c(...args)));
    case 4:
      return (...args) => a(b(c(d(...args))));
    case 5:
      return (...args) => a(b(c(d(e(...args)))));
    case 6:
      return (...args) => a(b(c(d(e(f(...args))))));
    case 7:
      return (...args) => a(b(c(d(e(f(g(...args)))))));
    case 8:
      return (...args) => a(b(c(d(e(f(g(h(...args))))))));
    case 9:
      return (...args) => a(b(c(d(e(f(g(h(i(...args)))))))));
    case 10:
    default:
      const fn = (...args) => a(b(c(d(e(f(g(h(i(j(...args))))))))));
      return fns.length === 10 ? fn : comp(fn, ...fns.slice(10));
  }
}
function compLeft(...fns) {
  return comp.apply(null, fns.reverse());
}
const compL = compLeft;
function compAsync(...fns) {
  return async (...args) => {
    let n = fns.length - 1;
    let res = fns[n](...args);
    while (n-- > 0) {
      res = await fns[n](res);
    }
    return res;
  };
}
export {
  comp,
  compAsync,
  compL,
  compLeft
};
