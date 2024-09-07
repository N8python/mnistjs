const doOnce = (fn, cache = /* @__PURE__ */ new Map()) => (x) => {
  if (!cache.has(x)) {
    cache.set(x, true);
    fn(x);
  }
};
export {
  doOnce
};
