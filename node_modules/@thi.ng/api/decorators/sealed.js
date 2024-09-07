const sealed = (constructor) => {
  Object.seal(constructor);
  Object.seal(constructor.prototype);
};
export {
  sealed
};
