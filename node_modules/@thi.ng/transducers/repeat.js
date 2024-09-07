function* repeat(x, n = Infinity) {
  while (n-- > 0) {
    yield x;
  }
}
export {
  repeat
};
