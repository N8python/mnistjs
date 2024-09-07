import { isArray } from "@thi.ng/checks/is-array";
import { isArrayLike } from "@thi.ng/checks/is-arraylike";
import { ensureIterable } from "./ensure-iterable.js";
const ensureArray = (x) => isArray(x) ? x : [...ensureIterable(x)];
const ensureArrayLike = (x) => isArrayLike(x) ? x : [...ensureIterable(x)];
export {
  ensureArray,
  ensureArrayLike
};
