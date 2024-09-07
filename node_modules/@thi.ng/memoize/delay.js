const delay = (body) => new Delay(body);
class Delay {
  value;
  body;
  realized;
  constructor(body) {
    this.body = body;
    this.realized = false;
  }
  deref() {
    if (!this.realized) {
      this.value = this.body();
      this.realized = true;
    }
    return this.value;
  }
  isRealized() {
    return this.realized;
  }
}
export {
  Delay,
  delay
};
