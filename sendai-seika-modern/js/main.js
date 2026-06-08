/* =========================================================
   仙台中央市場青果協同組合 — interactions
   Vanilla JS, no dependencies.
   ========================================================= */
(function () {
  "use strict";

  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Footer year ---------- */
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* ---------- Nav: scrolled state + mobile menu ---------- */
  const nav = document.getElementById("nav");
  const burger = document.getElementById("navBurger");
  const links = document.querySelector(".nav__links");

  const onScrollNav = () => {
    if (nav) nav.classList.toggle("scrolled", window.scrollY > 30);
  };
  onScrollNav();

  if (burger && links) {
    burger.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      burger.setAttribute("aria-expanded", String(open));
      burger.setAttribute("aria-label", open ? "メニューを閉じる" : "メニューを開く");
    });
    links.querySelectorAll("a").forEach((a) =>
      a.addEventListener("click", () => {
        links.classList.remove("open");
        burger.setAttribute("aria-expanded", "false");
      })
    );
  }

  /* ---------- Scroll progress bar ---------- */
  const progress = document.getElementById("scrollProgress");
  const updateProgress = () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    const pct = max > 0 ? (h.scrollTop || window.scrollY) / max : 0;
    if (progress) progress.style.width = (pct * 100).toFixed(2) + "%";
  };

  /* ---------- Reveal on scroll (fade in / out) ---------- */
  const revealEls = Array.from(document.querySelectorAll("[data-reveal]"));
  revealEls.forEach((el) => {
    const d = el.getAttribute("data-delay");
    if (d) el.style.transitionDelay = d + "ms";
  });

  if (prefersReduced) {
    revealEls.forEach((el) => el.classList.add("in"));
  } else {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const el = entry.target;
          if (entry.isIntersecting) {
            el.classList.add("in");
            el.classList.remove("out");
          } else {
            // fade back out only when leaving upward (scrolling further down past it
            // or scrolling up above it) for a living, breathing feel
            el.classList.remove("in");
            if (entry.boundingClientRect.top > 0) el.classList.remove("out");
            else el.classList.add("out");
          }
        });
      },
      { threshold: 0.18, rootMargin: "0px 0px -8% 0px" }
    );
    revealEls.forEach((el) => io.observe(el));
  }

  /* ---------- Animated number counters ---------- */
  const counters = Array.from(document.querySelectorAll("[data-count]"));
  const runCounter = (el) => {
    const target = parseFloat(el.getAttribute("data-count")) || 0;
    const suffix = el.getAttribute("data-suffix") || "";
    const dur = 1500;
    const start = performance.now();
    const tick = (now) => {
      const p = Math.min((now - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  if (counters.length) {
    const cio = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            runCounter(e.target);
            cio.unobserve(e.target);
          }
        });
      },
      { threshold: 0.5 }
    );
    counters.forEach((c) => cio.observe(c));
  }

  /* ---------- Parallax: hero 3D scene reacts to scroll & pointer ---------- */
  const stack = document.getElementById("crateStack");
  const heroScene = document.getElementById("heroScene");
  const warehouse = document.getElementById("warehouseScene");

  let pointerRY = 0;
  if (!prefersReduced && heroScene) {
    window.addEventListener(
      "pointermove",
      (e) => {
        const cx = window.innerWidth / 2;
        pointerRY = ((e.clientX - cx) / cx) * 10; // -10..10deg
        heroScene.style.transform = `rotateY(${pointerRY * 0.6}deg)`;
      },
      { passive: true }
    );
  }

  /* ---------- Tilt on glass cards ---------- */
  if (!prefersReduced) {
    document.querySelectorAll(".tilt").forEach((card) => {
      card.addEventListener("pointermove", (e) => {
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - 0.5;
        const py = (e.clientY - r.top) / r.height - 0.5;
        card.style.transform = `translateY(-8px) rotateX(${-py * 8}deg) rotateY(${px * 10}deg)`;
      });
      card.addEventListener("pointerleave", () => {
        card.style.transform = "";
      });
    });
  }

  /* ---------- Scroll-driven parallax on focus warehouse ---------- */
  const onScroll = () => {
    updateProgress();
    onScrollNav();
    if (!prefersReduced && warehouse) {
      const r = warehouse.getBoundingClientRect();
      const center = r.top + r.height / 2;
      const off = (center - window.innerHeight / 2) / window.innerHeight; // ~ -1..1
      warehouse.style.transform = `translateY(${off * -24}px)`;
    }
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* =========================================================
     Background canvas: drifting produce particles
     ========================================================= */
  const canvas = document.getElementById("bgCanvas");
  if (canvas && !prefersReduced) {
    const ctx = canvas.getContext("2d");
    let w, h, particles;
    const COLORS = ["#7cffb2", "#ffd23f", "#ff8a3d", "#ff5a5f", "#a855f7"];

    const resize = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
      const count = Math.min(48, Math.floor((w * h) / 32000));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: 1.5 + Math.random() * 3.5,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        c: COLORS[(Math.random() * COLORS.length) | 0],
        a: 0.2 + Math.random() * 0.4,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -10) p.x = w + 10;
        if (p.x > w + 10) p.x = -10;
        if (p.y < -10) p.y = h + 10;
        if (p.y > h + 10) p.y = -10;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.c;
        ctx.globalAlpha = p.a;
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(draw);
    };

    window.addEventListener("resize", resize);
    resize();
    draw();
  }
})();
