const align = (addr, size) => (size--, addr + size & ~size);
const isAligned = (addr, size) => !(addr & size - 1);
export {
  align,
  isAligned
};
