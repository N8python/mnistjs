const INV_MAX = 1 / 2 ** 32;
class ARandom {
  float(norm = 1) {
    return this.int() * INV_MAX * norm;
  }
  probability(p) {
    return this.float() < p;
  }
  norm(norm = 1) {
    return (this.int() * INV_MAX - 0.5) * 2 * norm;
  }
  normMinMax(min, max) {
    const x = this.minmax(min, max);
    return this.float() < 0.5 ? x : -x;
  }
  minmax(min, max) {
    return this.float() * (max - min) + min;
  }
  minmaxInt(min, max) {
    min |= 0;
    const range = (max | 0) - min;
    return range ? min + this.int() % range : min;
  }
  minmaxUint(min, max) {
    min >>>= 0;
    const range = (max >>> 0) - min;
    return range ? min + this.int() % range : min;
  }
}
export {
  ARandom
};
