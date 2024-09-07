function* repeatedly(fn, n = Infinity) {
  for (let i = 0; i < n; i++) {
    yield fn(i);
  }
}
export {
  repeatedly
};
