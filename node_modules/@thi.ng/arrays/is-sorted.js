import { compare } from "@thi.ng/compare/compare";
const isSorted = (arr, cmp = compare, start = 0, end = arr.length) => {
  let prev = arr[start];
  while (++start < end) {
    const curr = arr[start];
    if (cmp(prev, curr) > 0) return false;
    prev = curr;
  }
  return true;
};
export {
  isSorted
};
