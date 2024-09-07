import { repeat } from "./repeat.js";
const __nextTab = (x, tabSize) => Math.floor((x + tabSize) / tabSize) * tabSize;
const tabsToSpaces = (src, tabSize = 4) => src.split(/\r?\n/g).map((line) => tabsToSpacesLine(line, tabSize)).join("\n");
const tabsToSpacesLine = (line, tabSize = 4) => {
  let res = "";
  let words = line.split(/\t/g);
  let n = words.length - 1;
  for (let i = 0; i < n; i++) {
    const w = words[i];
    res += w;
    res += repeat(" ", __nextTab(res.length, tabSize) - res.length);
  }
  res += words[n];
  return res;
};
const spacesToTabs = (src, tabSize = 4) => src.split(/\r?\n/g).map((line) => spacesToTabsLine(line, tabSize)).join("\n");
const spacesToTabsLine = (line, tabSize = 4) => {
  const re = /\s{2,}/g;
  let i = 0;
  let res = "";
  let m;
  while (m = re.exec(line)) {
    const numSpaces = m[0].length;
    res += line.substring(i, m.index);
    i = m.index;
    const end = m.index + numSpaces;
    while (i < end) {
      const j = __nextTab(i, tabSize);
      if (j <= end) {
        res += "	";
        i = j;
      } else {
        res += repeat(" ", end - i);
        i = end;
      }
    }
    i = end;
  }
  return res + line.substring(i);
};
export {
  spacesToTabs,
  spacesToTabsLine,
  tabsToSpaces,
  tabsToSpacesLine
};
