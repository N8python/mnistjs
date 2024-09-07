const renamer = (kmap) => {
  const ks = Object.keys(kmap);
  const [a2, b2, c2] = ks;
  const [a1, b1, c1] = ks.map(
    (k) => kmap[k] === true ? k : kmap[k]
  );
  switch (ks.length) {
    case 3:
      return (x) => {
        const res = {};
        let v;
        (v = x[c1]) !== void 0 && (res[c2] = v);
        (v = x[b1]) !== void 0 && (res[b2] = v);
        (v = x[a1]) !== void 0 && (res[a2] = v);
        return res;
      };
    case 2:
      return (x) => {
        const res = {};
        let v;
        (v = x[b1]) !== void 0 && (res[b2] = v);
        (v = x[a1]) !== void 0 && (res[a2] = v);
        return res;
      };
    case 1:
      return (x) => {
        const res = {};
        let v;
        (v = x[a1]) !== void 0 && (res[a2] = v);
        return res;
      };
    default:
      return (x) => {
        let k, kk, v;
        const res = {};
        for (let i = ks.length - 1; i >= 0; i--) {
          k = ks[i];
          kk = kmap[k];
          (v = x[kk === true ? k : kk]) !== void 0 && (res[k] = v);
        }
        return res;
      };
  }
};
export {
  renamer
};
