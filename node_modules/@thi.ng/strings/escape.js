import { U16, U32 } from "./radix.js";
const ESCAPES = {
  0: "\0",
  b: "\b",
  t: "	",
  r: "\r",
  v: "\v",
  f: "\f",
  n: "\n",
  "'": "'",
  '"': '"',
  "\\": "\\"
};
const ESCAPES_REV = {
  0: "0",
  8: "b",
  9: "t",
  10: "n",
  11: "v",
  12: "f",
  13: "r",
  33: "'",
  34: '"',
  92: "\\"
};
const escape = (src) => src.replace(
  /[\0\b\t\n\v\f\r'"\\]/g,
  (x) => `\\${ESCAPES_REV[x.charCodeAt(0)]}`
).replace(/[\ud800-\udfff]{2}/g, (x) => `\\U${U32(x.codePointAt(0))}`).replace(/[^\u0020-\u007e]/g, (x) => `\\u${U16(x.charCodeAt(0))}`);
const unescape = (src) => src.replace(
  /\\u([0-9a-fA-F]{4})/g,
  (_, id) => String.fromCharCode(parseInt(id, 16))
).replace(
  /\\U([0-9a-fA-F]{8})/g,
  (_, id) => String.fromCodePoint(parseInt(id, 16))
).replace(/\\([0btnvfr'"\\])/g, (_, id) => ESCAPES[id]);
export {
  ESCAPES,
  ESCAPES_REV,
  escape,
  unescape
};
