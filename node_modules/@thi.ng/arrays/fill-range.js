const fillRange = (buf, index = 0, start = 0, end = buf.length, step = end > start ? 1 : -1) => {
  if (step > 0) {
    for (; start < end; start += step) buf[index++] = start;
  } else {
    for (; start > end; start += step) buf[index++] = start;
  }
  return buf;
};
export {
  fillRange
};
