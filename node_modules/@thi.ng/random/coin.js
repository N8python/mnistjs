import { SYSTEM } from "./system.js";
const coin = (rnd = SYSTEM) => rnd.float() < 0.5;
const fairCoin = (rnd = SYSTEM) => {
  let a, b;
  do {
    a = coin(rnd);
    b = coin(rnd);
  } while (a === b);
  return a;
};
export {
  coin,
  fairCoin
};
