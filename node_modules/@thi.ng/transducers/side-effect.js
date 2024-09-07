import { map } from "./map.js";
const sideEffect = (fn) => map((x) => (fn(x), x));
export {
  sideEffect
};
