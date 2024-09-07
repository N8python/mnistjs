import { repeat } from "./repeat.js";
const ruler = (width, major = 5, a = "|", b = "'") => repeat(a + repeat(b, major - 1), Math.ceil(width / major)).substring(
  0,
  width
);
const grid = (cols, rows, w, h, [a, b, c, d] = "+-| ") => {
  const major = ruler(cols * w, w, a, b) + a + "\n";
  const minor = ruler(cols * w, w, c, d) + c + "\n";
  return repeat(major + repeat(minor, h - 1), rows) + major;
};
export {
  grid,
  ruler
};
