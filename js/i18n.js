/* =========================================================
   Bilingual toggle (日本語 / English)
   Japanese is the default markup; English lives in data-en.
   Works on every page; choice persists via localStorage.
   ========================================================= */
(function () {
  "use strict";

  const KEY = "ssn-lang";
  const STR = {
    ja: { toggle: "EN", aria: "Switch to English" },
    en: { toggle: "日本語", aria: "日本語に切り替え" },
  };

  function apply(lang) {
    const isEn = lang === "en";
    document.documentElement.lang = isEn ? "en" : "ja";

    document.querySelectorAll("[data-en]").forEach((el) => {
      // Cache the original Japanese the first time we touch the node.
      if (el.dataset.ja === undefined) el.dataset.ja = el.innerHTML;
      el.innerHTML = isEn ? el.dataset.en : el.dataset.ja;
    });

    // Attribute translations (e.g. aria-label, alt) via data-en-attr="attr|English"
    document.querySelectorAll("[data-en-attr]").forEach((el) => {
      const [attr, val] = el.dataset.enAttr.split("|");
      if (!attr) return;
      const cacheKey = "ja_" + attr;
      if (el.dataset[cacheKey] === undefined)
        el.dataset[cacheKey] = el.getAttribute(attr) || "";
      el.setAttribute(attr, isEn ? val : el.dataset[cacheKey]);
    });

    // Page <title> via data-en-title on <html>
    const t = document.documentElement.dataset;
    if (t.enTitle) {
      if (t.jaTitle === undefined) t.jaTitle = document.title;
      document.title = isEn ? t.enTitle : t.jaTitle;
    }

    const btn = document.getElementById("langToggle");
    if (btn) {
      btn.textContent = STR[lang].toggle;
      btn.setAttribute("aria-label", STR[lang].aria);
    }
    try {
      localStorage.setItem(KEY, lang);
    } catch (e) {}
  }

  function init() {
    let lang = "ja";
    try {
      lang = localStorage.getItem(KEY) || "ja";
    } catch (e) {}
    apply(lang);

    const btn = document.getElementById("langToggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const next = document.documentElement.lang === "en" ? "ja" : "en";
        apply(next);
      });
    }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
