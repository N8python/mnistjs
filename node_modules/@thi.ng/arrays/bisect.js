function bisect(src, index = src.length >>> 1) {
  return [src.slice(0, index), src.slice(index)];
}
function bisectWith(src, pred) {
  const i = src.findIndex(pred);
  return i >= 0 ? bisect(src, i) : [src, src.slice(0, 0)];
}
export {
  bisect,
  bisectWith
};
