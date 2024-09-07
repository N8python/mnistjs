function* arrayIterator(buf, start = 0, end) {
  if (!buf) return;
  end === void 0 && (end = buf.length);
  const step = start <= end ? 1 : -1;
  for (; start !== end; start += step) {
    yield buf[start];
  }
}
export {
  arrayIterator
};
