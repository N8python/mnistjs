import { ARandom } from "./arandom.js";
import { DEFAULT_SEED_32 } from "./constants.js";
class Smush32 extends ARandom {
  buffer;
  constructor(seed = DEFAULT_SEED_32) {
    super();
    this.buffer = new Uint32Array([seed, 0]);
  }
  copy() {
    const gen = new Smush32();
    gen.buffer.set(this.buffer);
    return gen;
  }
  seed(s) {
    this.buffer.set([s, 0]);
    return this;
  }
  int() {
    const b = this.buffer;
    const m = 1540483477;
    const k = b[1]++ * m >>> 0;
    const s = b[0] = (k ^ k >> 24 ^ b[0] * m >>> 0) * m >>> 0;
    return (s ^ s >>> 13) >>> 0;
  }
}
export {
  Smush32
};
