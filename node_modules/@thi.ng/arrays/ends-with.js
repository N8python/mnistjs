import { equiv as _eq } from "@thi.ng/equiv";
const endsWith = (buf, needle, equiv = _eq) => {
  let i = buf.length;
  let j = needle.length;
  if (i < j) return false;
  while (--i, j-- > 0 && equiv(buf[i], needle[j])) {
  }
  return j < 0;
};
export {
  endsWith
};
