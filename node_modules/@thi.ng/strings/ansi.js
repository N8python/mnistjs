const RE = /\x1b\[[0-9;]+m/g;
const stripAnsi = (x) => x.replace(RE, "");
const isAnsi = (x) => /\x1b\[[0-9;]+m/i.test(x);
const isAnsiStart = (x) => /^\x1b\[[0-9;]+m/i.test(x);
const isAnsiEnd = (x) => /\x1b\[[0-9;]+m$/i.test(x);
const lengthAnsi = (x) => stripAnsi(x).length;
export {
  isAnsi,
  isAnsiEnd,
  isAnsiStart,
  lengthAnsi,
  stripAnsi
};
