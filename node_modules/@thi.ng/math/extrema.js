const isMinima = (a, b, c) => a > b && b < c;
const isMaxima = (a, b, c) => a < b && b > c;
const __index = (pred, values, from = 0, to = values.length) => {
  to--;
  for (let i = from + 1; i < to; i++) {
    if (pred(values[i - 1], values[i], values[i + 1])) {
      return i;
    }
  }
  return -1;
};
const minimaIndex = (values, from = 0, to = values.length) => __index(isMinima, values, from, to);
const maximaIndex = (values, from = 0, to = values.length) => __index(isMaxima, values, from, to);
function* indices(fn, vals, from = 0, to = vals.length) {
  while (from < to) {
    const i = fn(vals, from, to);
    if (i < 0) return;
    yield i;
    from = i + 1;
  }
}
const minimaIndices = (values, from = 0, to = values.length) => indices(minimaIndex, values, from, to);
const maximaIndices = (values, from = 0, to = values.length) => indices(minimaIndex, values, from, to);
export {
  isMaxima,
  isMinima,
  maximaIndex,
  maximaIndices,
  minimaIndex,
  minimaIndices
};
