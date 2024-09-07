function* iterate(fn, seed, num = Infinity) {
  for (let i = 1; i <= num; i++) {
    yield seed;
    seed = fn(seed, i);
  }
}
export {
  iterate
};
