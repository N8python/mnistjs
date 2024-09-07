const rotate = (buf, num) => {
  if (!(num = __distance(buf, num))) return buf;
  if (num < 0) {
    buf.push(...buf.splice(0, -num));
  } else {
    buf.unshift(...buf.splice(-num));
  }
  return buf;
};
const rotateTyped = (buf, num) => {
  if (!(num = __distance(buf, num))) return buf;
  if (num < 0) {
    const tmp = buf.slice(0, -num);
    buf.copyWithin(0, -num);
    buf.set(tmp, buf.length + num);
  } else if (num > 0) {
    const tmp = buf.slice(buf.length - num);
    buf.copyWithin(num, 0);
    buf.set(tmp, 0);
  }
  return buf;
};
const __distance = (buf, num) => buf.length ? (num | 0) % buf.length : 0;
export {
  rotate,
  rotateTyped
};
