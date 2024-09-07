import { memoizeJ } from "@thi.ng/memoize/memoizej";
const int = (x) => String(Math.trunc(x));
const intLocale = memoizeJ(
  (locale) => (x) => Math.trunc(x).toLocaleString(locale)
);
export {
  int,
  intLocale
};
