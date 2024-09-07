import { compare } from "@thi.ng/compare/compare";
import { compareNumAsc } from "@thi.ng/compare/numeric";
const binarySearch = (buf, x, key = (x2) => x2, cmp = compare, low = 0, high = buf.length - 1) => {
  const kx = key(x);
  while (low <= high) {
    const mid = low + high >>> 1;
    const c = cmp(key(buf[mid]), kx);
    if (c < 0) {
      low = mid + 1;
    } else if (c > 0) {
      high = mid - 1;
    } else {
      return mid;
    }
  }
  return -low - 1;
};
const binarySearchNumeric = (buf, x, cmp = compareNumAsc, low = 0, high = buf.length - 1) => {
  while (low <= high) {
    const mid = low + high >>> 1;
    const c = cmp(buf[mid], x);
    if (c < 0) {
      low = mid + 1;
    } else if (c > 0) {
      high = mid - 1;
    } else {
      return mid;
    }
  }
  return -low - 1;
};
const binarySearch2 = (buf, x) => {
  let idx = buf[1] <= x ? 1 : 0;
  return buf[idx] === x ? idx : buf[0] < x ? -idx - 2 : -1;
};
const binarySearch4 = (buf, x) => {
  let idx = buf[2] <= x ? 2 : 0;
  idx |= buf[idx + 1] <= x ? 1 : 0;
  return buf[idx] === x ? idx : buf[0] < x ? -idx - 2 : -1;
};
const binarySearch8 = (buf, x) => {
  let idx = buf[4] <= x ? 4 : 0;
  idx |= buf[idx + 2] <= x ? 2 : 0;
  idx |= buf[idx + 1] <= x ? 1 : 0;
  return buf[idx] === x ? idx : buf[0] < x ? -idx - 2 : -1;
};
const binarySearch16 = (buf, x) => {
  let idx = buf[8] <= x ? 8 : 0;
  idx |= buf[idx + 4] <= x ? 4 : 0;
  idx |= buf[idx + 2] <= x ? 2 : 0;
  idx |= buf[idx + 1] <= x ? 1 : 0;
  return buf[idx] === x ? idx : buf[0] < x ? -idx - 2 : -1;
};
const binarySearch32 = (buf, x) => {
  let idx = buf[16] <= x ? 16 : 0;
  idx |= buf[idx + 4] <= x ? 8 : 0;
  idx |= buf[idx + 4] <= x ? 4 : 0;
  idx |= buf[idx + 2] <= x ? 2 : 0;
  idx |= buf[idx + 1] <= x ? 1 : 0;
  return buf[idx] === x ? idx : buf[0] < x ? -idx - 2 : -1;
};
const bsLT = (i) => i < 0 ? -i - 2 : i - 1;
const bsLE = (i) => i < 0 ? -i - 2 : i;
const bsGT = (i, n) => (i = i < 0 ? -i - 1 : i + 1, i < n ? i : -1);
const bsGE = (i, n) => (i = i < 0 ? -i - 1 : i, i < n ? i : -1);
const bsEQ = (i) => i < 0 ? -1 : i;
export {
  binarySearch,
  binarySearch16,
  binarySearch2,
  binarySearch32,
  binarySearch4,
  binarySearch8,
  binarySearchNumeric,
  bsEQ,
  bsGE,
  bsGT,
  bsLE,
  bsLT
};
