function* curve(start, end, steps = 10, rate = 0.1) {
  const c = Math.exp(
    -Math.log((Math.abs(end - start) + rate) / rate) / steps
  );
  const offset = (start < end ? end + rate : end - rate) * (1 - c);
  steps > 0 && (yield start);
  for (let x = start; steps-- > 0; ) {
    yield x = offset + x * c;
  }
}
export {
  curve
};
