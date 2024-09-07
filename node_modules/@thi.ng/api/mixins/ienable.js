import { EVENT_DISABLE, EVENT_ENABLE } from "../api.js";
import { mixin } from "../mixin.js";
const IEnableMixin = mixin({
  _enabled: true,
  isEnabled() {
    return this._enabled;
  },
  enable() {
    $enable(this, true, EVENT_ENABLE);
  },
  disable() {
    $enable(this, false, EVENT_DISABLE);
  },
  toggle() {
    this._enabled ? this.disable() : this.enable();
    return this._enabled;
  }
});
const $enable = (target, state, id) => {
  target._enabled = state;
  if (target.notify) {
    target.notify({ id, target });
  }
};
export {
  IEnableMixin
};
