const CACHE = /* @__PURE__ */ Object.create(null);
const defOnce = (id, factory) => id in CACHE ? CACHE[id] : CACHE[id] = factory();
export {
  defOnce
};
