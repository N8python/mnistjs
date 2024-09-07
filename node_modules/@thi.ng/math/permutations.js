const factorial = (n) => {
  if (n < 0) throw new Error(`illegal argument: ${n}`);
  let res = 1;
  for (let i = 1; i <= n; i++) res *= i;
  return res;
};
const permutationsWithRep = (n, k) => n ** k;
const permutationsWithoutRep = (n, k) => factorial(n) / factorial(n - k);
const combinationsWithRep = (n, k) => factorial(n + k - 1) / (factorial(k) * factorial(n - 1));
const combinationsWithoutRep = (n, k) => factorial(n) / (factorial(k) * factorial(n - k));
export {
  combinationsWithRep,
  combinationsWithoutRep,
  factorial,
  permutationsWithRep,
  permutationsWithoutRep
};
