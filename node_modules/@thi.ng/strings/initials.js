const initials = (parts, mode = "u") => {
  const res = parts.map((x) => x[0]).join("");
  return mode === "u" ? res.toUpperCase() : mode === "l" ? res.toLowerCase() : res;
};
export {
  initials
};
