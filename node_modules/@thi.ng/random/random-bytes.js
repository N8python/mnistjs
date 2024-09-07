import { SYSTEM } from "./system.js";
const randomBytesFrom = (rnd, buf, start = 0, end = buf.length) => {
  for (let i = start; i < end; i++) {
    buf[i] = rnd.int() & 255;
  }
  return buf;
};
const randomBytes = typeof window !== "undefined" && window["crypto"] !== void 0 ? (buf, start = 0, end = buf.length) => (window.crypto.getRandomValues(buf.subarray(start, end)), buf) : (buf, start, end) => randomBytesFrom(SYSTEM, buf, start, end);
export {
  randomBytes,
  randomBytesFrom
};
