const swap = (arr, x, y) => {
  const t = arr[x];
  arr[x] = arr[y];
  arr[y] = t;
};
const multiSwap = (...arrays) => {
  const [b, c, d] = arrays;
  const n = arrays.length;
  switch (n) {
    case 0:
      return swap;
    case 1:
      return (a, x, y) => {
        swap(a, x, y);
        swap(b, x, y);
      };
    case 2:
      return (a, x, y) => {
        swap(a, x, y);
        swap(b, x, y);
        swap(c, x, y);
      };
    case 3:
      return (a, x, y) => {
        swap(a, x, y);
        swap(b, x, y);
        swap(c, x, y);
        swap(d, x, y);
      };
    default:
      return (a, x, y) => {
        swap(a, x, y);
        for (let i = n; i-- > 0; ) swap(arrays[i], x, y);
      };
  }
};
export {
  multiSwap,
  swap
};
