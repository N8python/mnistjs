const configurable = (state) => function(_, __, descriptor) {
  descriptor.configurable = state;
};
export {
  configurable
};
