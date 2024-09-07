import { sideEffect } from "./side-effect.js";
const trace = (prefix = "") => sideEffect((x) => console.log(prefix, x));
export {
  trace
};
