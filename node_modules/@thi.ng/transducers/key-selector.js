import { renamer } from "./renamer.js";
const keySelector = (keys) => renamer(keys.reduce((acc, x) => (acc[x] = x, acc), {}));
export {
  keySelector
};
