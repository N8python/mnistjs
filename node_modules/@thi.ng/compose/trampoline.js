const trampoline = (f) => {
  while (typeof f === "function") {
    f = f();
  }
  return f;
};
const trampolineAsync = async (f) => {
  f = await f;
  while (typeof f === "function") {
    f = await f();
  }
  return f;
};
export {
  trampoline,
  trampolineAsync
};
