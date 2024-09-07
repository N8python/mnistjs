const OPERATORS = {
  "=": eq,
  "!=": neq,
  "<": lt,
  "<=": lte,
  ">=": gte,
  ">": gt
};
const __ensure = (op) => {
  if (typeof op === "string") {
    if (op in OPERATORS) return OPERATORS[op];
    else throw new Error(`invalid operator: ${op}`);
  }
  return op;
};
const stringOp = (op, x) => {
  const impl = __ensure(op);
  return (y) => impl(String(y), x);
};
const numericOp = (op, x) => {
  const impl = __ensure(op);
  return (y) => typeof y === "number" && impl(y, x);
};
function eq(a, b) {
  return a === b;
}
function neq(a, b) {
  return a !== b;
}
function lt(a, b) {
  return a < b;
}
function lte(a, b) {
  return a <= b;
}
function gte(a, b) {
  return a >= b;
}
function gt(a, b) {
  return a > b;
}
export {
  OPERATORS,
  eq,
  gt,
  gte,
  lt,
  lte,
  neq,
  numericOp,
  stringOp
};
