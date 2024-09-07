import { int } from "./int.js";
const NAMED_DIGITS = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine"
];
const namedNumber = (names = NAMED_DIGITS) => (x) => x >= 0 && x < names.length ? names[x | 0] : int(x);
export {
  NAMED_DIGITS,
  namedNumber
};
