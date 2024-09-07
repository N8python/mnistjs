import { memoizeO } from "@thi.ng/memoize/memoizeo";
const trim = memoizeO((chars = " 	\n\r") => {
  chars = `(${chars.split("").map((x) => `\\${x}`).join("|")})`;
  const re = new RegExp(`(^${chars}+)|(${chars}+$)`, "g");
  return (x) => x.replace(re, "");
});
export {
  trim
};
