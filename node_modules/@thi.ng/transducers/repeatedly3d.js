function* repeatedly3d(fn, cols, rows, slices) {
  for (let z = 0; z < slices; z++) {
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        yield fn(x, y, z);
      }
    }
  }
}
export {
  repeatedly3d
};
