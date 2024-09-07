import { illegalArity } from "@thi.ng/errors/illegal-arity";
import { ensureTransducer } from "./ensure.js";
import { map } from "./map.js";
import { reduce, reduceRight } from "./reduce.js";
function transduce(...args) {
  return $transduce(transduce, reduce, args);
}
function transduceRight(...args) {
  return $transduce(transduceRight, reduceRight, args);
}
const $transduce = (tfn, rfn, args) => {
  let acc, src;
  switch (args.length) {
    case 4:
      src = args[3];
      acc = args[2];
      break;
    case 3:
      src = args[2];
      break;
    case 2:
      return map((x) => tfn(args[0], args[1], x));
    default:
      illegalArity(args.length);
  }
  return rfn(ensureTransducer(args[0])(args[1]), acc, src);
};
export {
  transduce,
  transduceRight
};
