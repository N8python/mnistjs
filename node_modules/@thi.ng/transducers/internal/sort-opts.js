import { compare } from "@thi.ng/compare/compare";
const __sortOpts = (opts) => ({
  key: (x) => x,
  compare,
  ...opts
});
export {
  __sortOpts
};
