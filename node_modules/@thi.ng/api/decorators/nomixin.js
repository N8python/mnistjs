const nomixin = (_, __, descriptor) => {
  descriptor.configurable = false;
};
export {
  nomixin
};
