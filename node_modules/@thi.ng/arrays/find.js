import { equiv as _equiv } from "@thi.ng/equiv";
const find = (buf, x, equiv = _equiv) => {
  const i = findIndex(buf, x, equiv);
  return i !== -1 ? buf[i] : void 0;
};
const findIndex = (buf, x, equiv = _equiv) => {
  for (let i = buf.length; i-- > 0; ) {
    if (equiv(x, buf[i])) return i;
  }
  return -1;
};
export {
  find,
  findIndex
};
