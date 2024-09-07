import { defError } from "@thi.ng/errors/deferror";
const utf8Length = (str) => {
  const n = str.length;
  let len = 0;
  for (let i = 0; i < n; ++i) {
    let u = str.charCodeAt(i);
    if (u >= 55296 && u < 917504) {
      u = 65536 + ((u & 1023) << 10) | str.charCodeAt(++i) & 1023;
    }
    len += u < 128 ? 1 : u < 2048 ? 2 : u < 65536 ? 3 : u < 2097152 ? 4 : u < 67108864 ? 5 : 6;
  }
  return len;
};
const utf8Decode = (buf, start, num) => {
  const end = start + num;
  let i = start;
  let result = "";
  let c;
  while (i < end) {
    c = buf[i++];
    if (c < 128) {
      result += String.fromCharCode(c);
    } else {
      if (c >= 192 && c < 224) {
        c = (c & 31) << 6 | buf[i++] & 63;
      } else if (c >= 224 && c < 240) {
        c = (c & 15) << 12 | (buf[i++] & 63) << 6 | buf[i++] & 63;
      } else if (c >= 240 && c < 248) {
        c = (c & 7) << 18 | (buf[i++] & 63) << 12 | (buf[i++] & 63) << 6 | buf[i++] & 63;
      } else __utf8Error();
      result += fromUtf8CodePoint(c);
    }
  }
  return result;
};
const utf8Encode = (src, capacity) => {
  const n = src.length;
  const buf = new Uint8Array(capacity || n << 2);
  let pos = 0;
  let c;
  for (let i = 0; i < n; i++) {
    c = src.charCodeAt(i);
    if (c < 128) {
      buf[pos++] = c;
    } else {
      if (c < 2048) {
        buf[pos++] = 192 | c >> 6;
      } else {
        if (c >= 55296 && c < 56320) {
          c = 65536 + ((c & 1023) << 10) + (src.charCodeAt(++i) & 1023);
          buf[pos++] = 240 | c >> 18;
          buf[pos++] = 128 | c >> 12 & 63;
        } else buf[pos++] = 224 | c >> 12;
        buf[pos++] = 128 | c >> 6 & 63;
      }
      buf[pos++] = 128 | c & 63;
    }
  }
  return buf.subarray(0, pos);
};
const fromUtf8CodePoint = (x) => {
  if (x < 65536) return String.fromCharCode(x);
  if (x < 1114112) {
    x -= 65536;
    return String.fromCharCode(55296 | x >>> 10, 56320 | x & 1023);
  }
  return __utf8Error(`invalid codepoint 0x${x.toString(16)}`);
};
const UTF8Error = defError(() => "UTF-8 error");
const __utf8Error = (msg) => {
  throw new UTF8Error(msg);
};
export {
  UTF8Error,
  fromUtf8CodePoint,
  utf8Decode,
  utf8Encode,
  utf8Length
};
