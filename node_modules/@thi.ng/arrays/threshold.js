const selectThresholdMin = (thresholds, defaultVal) => {
  const $thresholds = Object.entries(thresholds).map(([k, v]) => [+k, v]).sort((a, b) => a[0] - b[0]);
  return (x) => {
    const res = $thresholds.find((t) => x <= t[0]);
    return res ? res[1] : defaultVal;
  };
};
const selectThresholdMax = (thresholds, defaultVal) => {
  const $thresholds = Object.entries(thresholds).map(([k, v]) => [+k, v]).sort((a, b) => b[0] - a[0]);
  return (x) => {
    const res = $thresholds.find((t) => x >= t[0]);
    return res ? res[1] : defaultVal;
  };
};
export {
  selectThresholdMax,
  selectThresholdMin
};
