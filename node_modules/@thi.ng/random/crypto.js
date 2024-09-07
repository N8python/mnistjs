import { ARandom } from "./arandom.js";
import { randomBytes } from "./random-bytes.js";
class Crypto extends ARandom {
  buffer;
  u32;
  i;
  /**
   * @param size - buffer size in bytes (will be rounded to next multiple of 4)
   */
  constructor(size = 1024) {
    super();
    this.buffer = new Uint8Array(size + 3 & ~3);
    this.u32 = new Uint32Array(this.buffer.buffer);
    this.i = size >>> 2;
  }
  copy() {
    return new Crypto(this.buffer.length);
  }
  bytes() {
    return new Uint8Array(this.buffer.buffer);
  }
  int() {
    if (this.i >= this.u32.length) {
      randomBytes(this.buffer);
      this.i = 0;
    }
    return this.u32[this.i++];
  }
}
const CRYPTO = new Crypto();
export {
  CRYPTO,
  Crypto
};
