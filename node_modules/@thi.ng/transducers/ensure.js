import { implementsFunction } from "@thi.ng/checks/implements-function";
const ensureTransducer = (x) => implementsFunction(x, "xform") ? x.xform() : x;
export {
  ensureTransducer
};
