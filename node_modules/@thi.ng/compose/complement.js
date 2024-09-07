function complement(f) {
  return (...args) => !f(...args);
}
export {
  complement
};
