import { ensureArray } from "@thi.ng/arrays/ensure-array";
import { illegalArgs } from "@thi.ng/errors/illegal-arguments";
import { inRange } from "@thi.ng/math/interval";
function* wrapSides(src, numLeft = 1, numRight = numLeft) {
  const _src = ensureArray(src);
  !(inRange(numLeft, 0, _src.length) && inRange(numRight, 0, _src.length)) && illegalArgs(`allowed wrap range: [0..${_src.length}]`);
  if (numLeft > 0) {
    for (let m = _src.length, i = m - numLeft; i < m; i++) {
      yield _src[i];
    }
  }
  yield* _src;
  if (numRight > 0) {
    for (let i = 0; i < numRight; i++) {
      yield _src[i];
    }
  }
}
export {
  wrapSides
};
