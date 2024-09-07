import { compare } from "@thi.ng/compare/compare";
function pushSort(cmp = compare, src) {
  return src ? [...src].sort(cmp) : [
    () => [],
    (acc) => acc.sort(cmp),
    (acc, x) => (acc.push(x), acc)
  ];
}
export {
  pushSort
};
