const promisify = (fn) => new Promise(
  (resolve, reject) => fn((err, result) => err != null ? reject(err) : resolve(result))
);
export {
  promisify
};
