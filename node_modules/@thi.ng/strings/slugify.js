const SRC = "\xE0\xE1\xE4\xE2\xE3\xE5\xE8\xE9\xEB\xEA\xEC\xED\xEF\xEE\xF2\xF3\xF6\xF4\xF9\xFA\xFC\xFB\xF1\xE7\xDF\xFF\u0153\xE6\u0155\u015B\u0144\u1E55\u1E83\u01F5\u01F9\u1E3F\u01D8\u1E8D\u017A\u1E27\xB7/_,:;";
const DEST = "aaaaaaeeeeiiiioooouuuuncsyoarsnpwgnmuxzh------";
const RE = new RegExp(SRC.split("").join("|"), "g");
const slugify = (str) => {
  return str.toLowerCase().replace(/\s+/g, "-").replace(RE, (c) => DEST[SRC.indexOf(c)]).replace(/&+/g, "-and-").replace(/[^\w-]+/g, "").replace(/\p{Emoji_Presentation}/gu, "").replace(/-{2,}/g, "-").replace(/(^-+)|(-+$)/g, "");
};
const slugifyGH = (str) => {
  return str.toLowerCase().replace(
    /[!"#$%&'()*+,./:;<=>?@[\\\]^`{|}~\u0000-\u001f\u2000-\u206f\u2700-\u27bf\u2e00-\u2e7f]/g,
    ""
  ).replace(/\p{Emoji_Presentation}/gu, "").replace(/\s/g, "-");
};
export {
  slugify,
  slugifyGH
};
