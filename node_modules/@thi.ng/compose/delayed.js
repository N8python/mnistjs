const delayed = (x, t) => new Promise((resolve) => setTimeout(() => resolve(x), t));
export {
  delayed
};
