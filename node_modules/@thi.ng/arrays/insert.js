const insert = (buf, x, i, k = Infinity) => i < 0 || i >= k || k < 1 ? buf : insertUnsafe(buf, x, i, k);
const insertUnsafe = (buf, x, i, k = Infinity) => {
  let j = buf.length < k ? buf.length + 1 : k;
  for (; --j > i; ) buf[j] = buf[j - 1];
  buf[i] = x;
  return buf;
};
export {
  insert,
  insertUnsafe
};
