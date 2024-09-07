import { compare } from "@thi.ng/compare/compare";
function* sortedKeys(x, cmp = compare) {
  yield* Object.keys(x).sort(cmp);
}
export {
  sortedKeys
};
