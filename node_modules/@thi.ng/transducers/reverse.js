import { ensureArray } from "@thi.ng/arrays/ensure-array";
function* reverse(input) {
  const _input = ensureArray(input);
  let n = _input.length;
  while (n-- > 0) {
    yield _input[n];
  }
}
export {
  reverse
};
