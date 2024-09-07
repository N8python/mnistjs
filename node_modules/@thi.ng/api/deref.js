const isDeref = (x) => x != null && typeof x["deref"] === "function";
const deref = (x) => isDeref(x) ? x.deref() : x;
export {
  deref,
  isDeref
};
