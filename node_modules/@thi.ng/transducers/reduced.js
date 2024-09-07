class Reduced {
  value;
  constructor(val) {
    this.value = val;
  }
  deref() {
    return this.value;
  }
}
const reduced = (x) => new Reduced(x);
const isReduced = (x) => x instanceof Reduced;
const ensureReduced = (x) => x instanceof Reduced ? x : new Reduced(x);
const unreduced = (x) => x instanceof Reduced ? x.deref() : x;
export {
  Reduced,
  ensureReduced,
  isReduced,
  reduced,
  unreduced
};
