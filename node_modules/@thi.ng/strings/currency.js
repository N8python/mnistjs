import { memoizeJ } from "@thi.ng/memoize/memoizej";
const currency = memoizeJ((sym, pre = true, prec = 2) => {
  const ff = (x) => x.toFixed(prec);
  return pre ? (x) => sym + ff(x) : (x) => ff(x) + sym;
});
const chf = currency("CHF ");
const eur = currency("\u20AC");
const gbp = currency("\xA3");
const usd = currency("$");
const yen = currency("\xA5");
export {
  chf,
  currency,
  eur,
  gbp,
  usd,
  yen
};
