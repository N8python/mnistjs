import { equiv as _eq } from "@thi.ng/equiv";
const fuzzyMatch = (domain, query, equiv = _eq) => {
  const nd = domain.length;
  const nq = query.length;
  if (nq > nd) {
    return false;
  }
  if (nq === nd) {
    return equiv(query, domain);
  }
  next: for (let i = 0, j = 0; i < nq; i++) {
    const q = query[i];
    while (j < nd) {
      if (equiv(domain[j++], q)) {
        continue next;
      }
    }
    return false;
  }
  return true;
};
export {
  fuzzyMatch
};
