function* split(src, delim = /\r?\n/g, includeDelim = false) {
  let i = 0;
  const n = src.length;
  const include = ~~includeDelim;
  let re;
  if (typeof delim === "string") {
    re = new RegExp(delim, "g");
  } else if (!delim.flags.includes("g")) {
    re = new RegExp(delim, delim.flags + "g");
  } else {
    re = delim;
  }
  for (; i < n; ) {
    const m = re.exec(src);
    if (!m) {
      yield src.substring(i);
      return;
    }
    const len = m[0].length;
    yield src.substring(i, m.index + include * len);
    i = m.index + len;
  }
}
export {
  split
};
