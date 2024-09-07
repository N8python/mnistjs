function* primesUntil(x) {
  if (x < 1) return;
  yield 1;
  const sieve = [];
  const max = Math.sqrt(x) | 0;
  for (let i = 2; i <= x; i++) {
    if (!sieve[i]) {
      yield i;
      __updateSieve(sieve, i, x, max);
    }
  }
}
const nearestPrime = (x) => {
  if (x < 1) return -1;
  let prime = 1;
  const sieve = [];
  const max = Math.sqrt(x) | 0;
  for (let i = 2; i <= x; i++) {
    if (!sieve[i]) {
      prime = i;
      __updateSieve(sieve, i, x, max);
    }
  }
  return prime;
};
const __updateSieve = (sieve, i, x, max) => {
  if (i <= max) {
    for (let j = i * i; j <= x; j += i) sieve[j] = true;
  }
};
export {
  nearestPrime,
  primesUntil
};
