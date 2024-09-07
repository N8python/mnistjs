import { isArray } from "@thi.ng/checks/is-array";
import { isString } from "@thi.ng/checks/is-string";
import { concat } from "./concat.js";
import { reverse } from "./reverse.js";
import { str } from "./str.js";
import { symmetric } from "./symmetric.js";
function palindrome(x) {
  return isString(x) ? str("", concat([x], reverse(x))) : isArray(x) ? x.concat(x.slice().reverse()) : symmetric(x);
}
export {
  palindrome
};
