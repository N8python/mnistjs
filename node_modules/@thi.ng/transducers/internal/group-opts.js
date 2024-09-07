import { push } from "../push.js";
const __groupByOpts = (opts) => ({
  key: (x) => x,
  group: push(),
  ...opts
});
export {
  __groupByOpts
};
