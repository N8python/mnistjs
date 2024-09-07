const computeCursorPos = (str, pos, delim = "\n", offset = [1, 1]) => {
  if (!str.length) return [1, 1];
  pos = Math.min(Math.max(0, pos), str.length);
  const lines = str.split(delim);
  const n = lines.length;
  for (let i = 0; i < n; i++) {
    const l = lines[i];
    if (pos <= l.length) return [i + offset[0], pos + offset[1]];
    pos -= l.length + 1;
  }
  return [n + offset[0] - 1, offset[1]];
};
export {
  computeCursorPos
};
