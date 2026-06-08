/* =========================================================
   Bilingual toggle (日本語 / English)
   Japanese is the default markup; English lives in data-en.
   Works on every page; choice persists via localStorage.

   Robust by design: each node's swap is isolated in try/catch so a
   single problematic element can never stop the rest of the page
   from translating. Originals are cached in a WeakMap (not in
   data-* attributes) to avoid any attribute-escaping pitfalls.
   ========================================================= */
(function () {
  "use strict";

  var KEY = "ssn-lang";
  var STR = {
    ja: { toggle: "EN", aria: "Switch to English" },
    en: { toggle: "日本語", aria: "日本語に切り替え" },
  };

  var htmlOriginals = new WeakMap(); // el -> original (Japanese) innerHTML
  var attrOriginals = new WeakMap(); // el -> { attrName: originalValue }

  function apply(lang) {
    var isEn = lang === "en";
    document.documentElement.lang = isEn ? "en" : "ja";

    // 1) Text / inner HTML
    var nodes = document.querySelectorAll("[data-en]");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      try {
        if (!htmlOriginals.has(el)) htmlOriginals.set(el, el.innerHTML);
        el.innerHTML = isEn
          ? el.getAttribute("data-en") || ""
          : htmlOriginals.get(el);
      } catch (e) {
        /* keep going so the rest of the page still translates */
      }
    }

    // 2) Attribute translations:  data-en-attr="attr|English value"
    var attrNodes = document.querySelectorAll("[data-en-attr]");
    for (var j = 0; j < attrNodes.length; j++) {
      var ael = attrNodes[j];
      try {
        var spec = ael.getAttribute("data-en-attr") || "";
        var sep = spec.indexOf("|");
        if (sep < 0) continue;
        var attr = spec.slice(0, sep);
        var val = spec.slice(sep + 1);
        var store = attrOriginals.get(ael) || {};
        if (!(attr in store)) {
          store[attr] = ael.getAttribute(attr) || "";
          attrOriginals.set(ael, store);
        }
        ael.setAttribute(attr, isEn ? val : store[attr]);
      } catch (e) {}
    }

    // 3) Document title:  <html data-en-title="...">
    try {
      var dt = document.documentElement;
      var enTitle = dt.getAttribute("data-en-title");
      if (enTitle) {
        if (!htmlOriginals.has(dt)) htmlOriginals.set(dt, document.title);
        document.title = isEn ? enTitle : htmlOriginals.get(dt);
      }
    } catch (e) {}

    // 4) Toggle button label
    var btn = document.getElementById("langToggle");
    if (btn) {
      btn.textContent = STR[lang].toggle;
      btn.setAttribute("aria-label", STR[lang].aria);
    }

    try { localStorage.setItem(KEY, lang); } catch (e) {}
  }

  function init() {
    // Always start in Japanese on every page load (default language).
    apply("ja");

    var btn = document.getElementById("langToggle");
    if (btn) {
      btn.addEventListener("click", function () {
        apply(document.documentElement.lang === "en" ? "ja" : "en");
      });
    }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
