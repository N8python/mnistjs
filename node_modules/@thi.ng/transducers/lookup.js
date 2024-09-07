const lookup1d = (src) => (i) => src[i];
const lookup2d = (src, width) => (i) => src[i[0] + i[1] * width];
const lookup3d = (src, width, height) => {
  const stridez = width * height;
  return (i) => src[i[0] + i[1] * width + i[2] * stridez];
};
export {
  lookup1d,
  lookup2d,
  lookup3d
};
