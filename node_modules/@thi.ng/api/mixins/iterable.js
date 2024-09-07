import { mixin } from "../mixin.js";
const iterable = (prop) => mixin({
  *[Symbol.iterator]() {
    yield* this[prop];
  }
});
export {
  iterable
};
