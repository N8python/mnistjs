const argMin = (buf, min = Infinity, pred = (a, b) => a < b) => {
  let id = -1;
  for (let i = 0, n = buf.length; i < n; i++) {
    const x = buf[i];
    if (pred(x, min)) {
      min = x;
      id = i;
    }
  }
  return id;
};
const argMax = (items, min = -Infinity, pred = (a, b) => a > b) => argMin(items, min, pred);
export {
  argMax,
  argMin
};
