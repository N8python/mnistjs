const maybeParseInt = (x, defaultVal = 0, radix = 10) => {
  const n = parseInt(x, radix);
  return isNaN(n) ? defaultVal : n;
};
const maybeParseFloat = (x, defaultVal = 0) => {
  const n = parseFloat(x);
  return isNaN(n) ? defaultVal : n;
};
export {
  maybeParseFloat,
  maybeParseInt
};
