import { reducer } from "./reduce.js";
const pushCopy = () => reducer(
  () => [],
  (acc, x) => acc.concat([x])
);
export {
  pushCopy
};
