const BOUNDARY = /(\D\d|\d\D)/;
const compareLex = (a, b) => {
  const ma = BOUNDARY.exec(a);
  const mb = BOUNDARY.exec(b);
  return ma && mb ? __compare2(
    __maybeNum(a.substring(0, ma.index + 1)),
    __maybeNum(a.substring(ma.index + 1)),
    __maybeNum(b.substring(0, mb.index + 1)),
    __maybeNum(b.substring(mb.index + 1))
  ) : a < b ? -1 : a > b ? 1 : 0;
};
const __compare2 = (a1, a2, b1, b2) => {
  const na = typeof a1 === "number";
  const nb = typeof b1 === "number";
  if (na !== nb) {
    if (na) a1 += " ";
    else a2 += " ";
    if (nb) b1 += " ";
    else b2 += " ";
  }
  return a1 === b1 ? a2 === b2 ? 0 : a2 < b2 ? -2 : 2 : a1 < b1 ? -1 : 1;
};
const __maybeNum = (x) => {
  const id = x.charCodeAt(0);
  return id >= 48 && id <= 57 ? +x : x;
};
export {
  compareLex
};
