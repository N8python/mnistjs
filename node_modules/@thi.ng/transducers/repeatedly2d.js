function* repeatedly2d(fn, cols, rows) {
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      yield fn(x, y);
    }
  }
}
export {
  repeatedly2d
};
