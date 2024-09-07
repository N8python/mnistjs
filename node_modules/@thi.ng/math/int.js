const M8 = 255;
const M16 = 65535;
const signExtend8 = (a) => (a &= M8, a & 128 ? a | ~M8 : a);
const signExtend16 = (a) => (a &= M16, a & 32768 ? a | ~M16 : a);
const addi8 = (a, b) => signExtend8((a | 0) + (b | 0));
const divi8 = (a, b) => signExtend8((a | 0) / (b | 0));
const muli8 = (a, b) => signExtend8((a | 0) * (b | 0));
const subi8 = (a, b) => signExtend8((a | 0) - (b | 0));
const andi8 = (a, b) => signExtend8((a | 0) & (b | 0));
const ori8 = (a, b) => signExtend8(a | 0 | (b | 0));
const xori8 = (a, b) => signExtend8((a | 0) ^ (b | 0));
const noti8 = (a) => signExtend8(~a);
const lshifti8 = (a, b) => signExtend8((a | 0) << (b | 0));
const rshifti8 = (a, b) => signExtend8((a | 0) >> (b | 0));
const addi16 = (a, b) => signExtend16((a | 0) + (b | 0));
const divi16 = (a, b) => signExtend16((a | 0) / (b | 0));
const muli16 = (a, b) => signExtend16((a | 0) * (b | 0));
const subi16 = (a, b) => signExtend16((a | 0) - (b | 0));
const andi16 = (a, b) => signExtend16((a | 0) & (b | 0));
const ori16 = (a, b) => signExtend16(a | 0 | (b | 0));
const xori16 = (a, b) => signExtend16((a | 0) ^ (b | 0));
const noti16 = (a) => signExtend16(~a);
const lshifti16 = (a, b) => signExtend16((a | 0) << (b | 0));
const rshifti16 = (a, b) => signExtend16((a | 0) >> (b | 0));
const addi32 = (a, b) => (a | 0) + (b | 0) | 0;
const divi32 = (a, b) => (a | 0) / (b | 0) | 0;
const muli32 = (a, b) => (a | 0) * (b | 0) | 0;
const subi32 = (a, b) => (a | 0) - (b | 0) | 0;
const andi32 = (a, b) => (a | 0) & (b | 0);
const ori32 = (a, b) => a | 0 | (b | 0);
const xori32 = (a, b) => (a | 0) ^ (b | 0);
const lshifti32 = (a, b) => (a | 0) << (b | 0);
const rshifti32 = (a, b) => (a | 0) >> (b | 0);
const noti32 = (a) => ~a;
const addu8 = (a, b) => (a & M8) + (b & M8) & M8;
const divu8 = (a, b) => (a & M8) / (b & M8) & M8;
const mulu8 = (a, b) => (a & M8) * (b & M8) & M8;
const subu8 = (a, b) => (a & M8) - (b & M8) & M8;
const andu8 = (a, b) => a & M8 & (b & M8) & M8;
const oru8 = (a, b) => (a & M8 | b & M8) & M8;
const xoru8 = (a, b) => (a & M8 ^ b & M8) & M8;
const notu8 = (a) => ~a & M8;
const lshiftu8 = (a, b) => (a & M8) << (b & M8) & M8;
const rshiftu8 = (a, b) => (a & M8) >>> (b & M8) & M8;
const addu16 = (a, b) => (a & M16) + (b & M16) & M16;
const divu16 = (a, b) => (a & M16) / (b & M16) & M16;
const mulu16 = (a, b) => (a & M16) * (b & M16) & M16;
const subu16 = (a, b) => (a & M16) - (b & M16) & M16;
const andu16 = (a, b) => a & M16 & (b & M16) & M16;
const oru16 = (a, b) => (a & M16 | b & M16) & M16;
const xoru16 = (a, b) => (a & M16 ^ b & M16) & M16;
const notu16 = (a) => ~a & M16;
const lshiftu16 = (a, b) => (a & M16) << (b & M16) & M16;
const rshiftu16 = (a, b) => (a & M16) >>> (b & M16) & M16;
const addu32 = (a, b) => (a >>> 0) + (b >>> 0) >>> 0;
const divu32 = (a, b) => (a >>> 0) / (b >>> 0) >>> 0;
const mulu32 = (a, b) => (a >>> 0) * (b >>> 0) >>> 0;
const subu32 = (a, b) => (a >>> 0) - (b >>> 0) >>> 0;
const andu32 = (a, b) => (a >>> 0 & b >>> 0) >>> 0;
const oru32 = (a, b) => (a >>> 0 | b >>> 0) >>> 0;
const xoru32 = (a, b) => (a >>> 0 ^ b >>> 0) >>> 0;
const notu32 = (a) => ~a >>> 0;
const lshiftu32 = (a, b) => a >>> 0 << (b >>> 0) >>> 0;
const rshiftu32 = (a, b) => a >>> 0 >>> (b >>> 0) >>> 0;
export {
  addi16,
  addi32,
  addi8,
  addu16,
  addu32,
  addu8,
  andi16,
  andi32,
  andi8,
  andu16,
  andu32,
  andu8,
  divi16,
  divi32,
  divi8,
  divu16,
  divu32,
  divu8,
  lshifti16,
  lshifti32,
  lshifti8,
  lshiftu16,
  lshiftu32,
  lshiftu8,
  muli16,
  muli32,
  muli8,
  mulu16,
  mulu32,
  mulu8,
  noti16,
  noti32,
  noti8,
  notu16,
  notu32,
  notu8,
  ori16,
  ori32,
  ori8,
  oru16,
  oru32,
  oru8,
  rshifti16,
  rshifti32,
  rshifti8,
  rshiftu16,
  rshiftu32,
  rshiftu8,
  signExtend16,
  signExtend8,
  subi16,
  subi32,
  subi8,
  subu16,
  subu32,
  subu8,
  xori16,
  xori32,
  xori8,
  xoru16,
  xoru32,
  xoru8
};
