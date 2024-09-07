import { reducer } from "./reduce.js";
function str(sep, src) {
  sep = sep || "";
  let first = true;
  return src ? [...src].join(sep) : reducer(
    () => "",
    (acc, x) => (acc = first ? acc + x : acc + sep + x, first = false, acc)
  );
}
export {
  str
};
