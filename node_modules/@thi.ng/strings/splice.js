import { illegalArgs } from "@thi.ng/errors/illegal-arguments";
const splice = (src, insert, from, to = from) => {
  if (from < 0) {
    from += src.length;
  }
  if (to < 0) {
    to += src.length;
  }
  if (from > to) {
    illegalArgs("'from' index must be <= 'to'");
  }
  to = Math.max(to, 0);
  return from <= 0 ? insert + src.substring(to) : from >= src.length ? src + insert : src.substring(0, from) + insert + src.substring(to);
};
export {
  splice
};
