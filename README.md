# 仙台中央市場青果協同組合 — Modern site (concept redesign)

A fresh, modern, responsive redesign concept for **仙台中央市場青果協同組合
(Sendai Central Market Produce Cooperative)**, focused on the
**貯蔵保管・場内運搬事業 (Storage & In-Market Transport)** page —
originally `business4.html` on <https://www.ssn.or.jp>.

## ✨ What's inside
- **Full 3D artwork (no libraries)** — a rotating stack of produce crates and an
  isometric cold-storage warehouse, built entirely with CSS 3D transforms.
- **Scroll animations** — fade-in / fade-out reveals, scroll progress bar,
  parallax on the 3D scenes, animated number counters.
- **Fresh imagery** — generated particle field, aurora gradients, glassmorphism.
- **Fully responsive** — desktop → tablet → mobile, with a mobile menu.
- **Accessible** — respects `prefers-reduced-motion`, semantic landmarks.
- **Zero build step & zero dependencies** — just open the file.

## 📄 Pages
- `index.html` — home (hero 3D crates, business4 focus, stats, contact)
- `company.html` — 組合概要 (overview + profile table)
- `delivery.html` — 共同配送事業 (with 3D truck)
- `fuel.html` — 給油事業 (with history timeline)
- `members.html` — 組合員一覧 (member companies)

Every page has a **EN / 日本語 toggle** (top-right of the nav); the choice is
remembered across pages via `localStorage`.

## ▶️ How to open (at home)
Just double-click **`index.html`** — it runs in any modern browser, fully offline.

> Internet only enhances it (Google Fonts); it degrades gracefully to system
> fonts offline. No npm, no bundler, no server required. To serve it instead:
> `python3 -m http.server` then open <http://localhost:8000>.

## 📱 Live site
Published via GitHub Pages from `main` (`.github/workflows/static.yml`):

**https://johnpiere.github.io/DartsMovementTracking/**

## 📁 Structure
```
.
├── index.html            ← the site (home)
├── company.html          ← 組合概要
├── delivery.html         ← 共同配送事業
├── fuel.html             ← 給油事業
├── members.html          ← 組合員一覧
├── css/styles.css        ← styles, 3D, animations
├── js/main.js            ← scroll motion, parallax, counters, canvas
├── js/i18n.js            ← EN / 日本語 toggle
├── assets/favicon.svg
└── source-export/        ← reconstructed source content + scrape notes
    ├── SOURCE_NOTES.md   ← why original images couldn't be fetched + how to mirror
    └── content.md        ← reconstructed content map of ssn.or.jp
```

> The previous darts-tracker software that lived in this repo has been removed
> in favour of this site (it remains on the owner's local backup).

## ⚠️ Note on the original assets
The original site's text and photos could **not** be downloaded from the build
environment because its network policy blocks the `ssn.or.jp` host. The content
on this redesign was **reconstructed from web search** and should be treated as
representative — see `source-export/SOURCE_NOTES.md` for how to mirror the real
assets from your home machine (`wget -mkEpnp https://www.ssn.or.jp/`).
