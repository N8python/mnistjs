const format = (fmt, ...args) => {
  const acc = [];
  for (let i = 0, j = 0, n = fmt.length; i < n; i++) {
    const f = fmt[i];
    const t = typeof f;
    acc.push(
      t === "function" ? f(args[j++]) : t === "object" ? f[args[j++]] : f
    );
  }
  return acc.join("");
};
const defFormat = (fmt) => (...args) => format(fmt, ...args);
const ignore = (_) => "";
const str = (x) => String(x);
export {
  defFormat,
  format,
  ignore,
  str
};
