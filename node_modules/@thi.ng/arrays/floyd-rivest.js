import { compare } from "@thi.ng/compare/compare";
import { swap } from "./swap.js";
function floydRivest(buf, k = 1, cmp = compare, left = 0, right = buf.length - 1) {
  while (right > left) {
    if (right - left > 600) {
      const n = right - left + 1;
      const i2 = k - left + 1;
      const z = Math.log(n);
      const s = 0.5 * Math.exp(z * (2 / 3));
      const sd = 0.5 * Math.sqrt(z * s * ((n - s) / n)) * Math.sign(i2 - n / 2);
      floydRivest(
        buf,
        k,
        cmp,
        Math.max(left, k - i2 * s / n + sd | 0),
        Math.min(right, k + (n - i2) * s / n + sd | 0)
      );
    }
    const t = buf[k];
    let i = left;
    let j = right;
    swap(buf, left, k);
    if (cmp(buf[right], t) > 0) swap(buf, right, left);
    while (i < j) {
      swap(buf, i, j);
      i++;
      j--;
      while (compare(buf[i], t) < 0) i++;
      while (compare(buf[j], t) > 0) j--;
    }
    if (compare(buf[left], t) === 0) {
      swap(buf, left, j);
    } else {
      j++;
      swap(buf, j, right);
    }
    if (j <= k) left = j + 1;
    if (k <= j) right = j - 1;
  }
  return buf;
}
export {
  floydRivest
};
