function* charRange(from, to) {
  let i = typeof from === "string" ? from.charCodeAt(0) : from;
  const end = typeof to === "string" ? to.charCodeAt(0) : to;
  if (i <= end) {
    for (; i <= end; i++) {
      yield String.fromCharCode(i);
    }
  } else {
    for (; i >= end; i--) {
      yield String.fromCharCode(i);
    }
  }
}
export {
  charRange
};
