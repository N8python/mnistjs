import { maskL } from "./mask.js";
const bitNot = (x) => ~x;
const bitAnd = (a, b) => a & b;
const bitNand = (a, b) => ~(a & b);
const bitOr = (a, b) => a | b;
const bitNor = (a, b) => ~(a | b);
const bitXor = (a, b) => a ^ b;
const bitXnor = (a, b) => ~(a ^ b);
const bitImply = (a, b) => ~a | b;
const bitAoi21 = (a, b, c) => ~(a | b & c);
const bitOai21 = (a, b, c) => ~(a & (b | c));
const bitAoi22 = (a, b, c, d) => ~(a & b | c & d);
const bitOai22 = (a, b, c, d) => ~((a | b) & (c | d));
const bitMux = (a, b, s) => (a & ~s | b & s) >>> 0;
const bitDemux = (a, b, s) => [
  (a & ~s) >>> 0,
  (b & s) >>> 0
];
const bitNotM = (n, x) => maskL(n, ~x);
const bitAndM = (n, a, b) => maskL(n, a & b);
const bitNandM = (n, a, b) => maskL(n, ~(a & b));
const bitOrM = (n, a, b) => maskL(n, a | b);
const bitNorM = (n, a, b) => maskL(n, ~(a | b));
const bitXorM = (n, a, b) => maskL(n, a ^ b);
const bitXnorM = (n, a, b) => maskL(n, ~(a ^ b));
const bitImplyM = (n, a, b) => maskL(n, ~a | b);
const bitAoi21M = (n, a, b, c) => maskL(n, ~(a | b & c));
const bitOai21M = (n, a, b, c) => maskL(n, ~(a & (b | c)));
const bitAoi22M = (n, a, b, c, d) => maskL(n, ~(a & b | c & d));
const bitOai22M = (n, a, b, c, d) => maskL(n, ~((a | b) & (c | d)));
const bitMuxM = (n, a, b, s) => maskL(n, a & ~s | b & s);
const bitDemuxM = (n, a, b, s) => [
  maskL(n, a & ~s),
  maskL(n, b & s)
];
export {
  bitAnd,
  bitAndM,
  bitAoi21,
  bitAoi21M,
  bitAoi22,
  bitAoi22M,
  bitDemux,
  bitDemuxM,
  bitImply,
  bitImplyM,
  bitMux,
  bitMuxM,
  bitNand,
  bitNandM,
  bitNor,
  bitNorM,
  bitNot,
  bitNotM,
  bitOai21,
  bitOai21M,
  bitOai22,
  bitOai22M,
  bitOr,
  bitOrM,
  bitXnor,
  bitXnorM,
  bitXor,
  bitXorM
};
