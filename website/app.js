/* Cordi Lab storefront: hash router, cart, checkout, mascot */

"use strict";

/* ---------- Catalog ---------- */

const PRODUCTS = {
  "soft-landing-single": {
    id: "soft-landing-single",
    name: "Soft Landing Blind Box",
    sub: "Single blind box",
    price: 7.99,
    stripe: "https://buy.stripe.com/3cIbIUf7p0rS5sV8jc4Rq01",
  },
  "soft-landing-set": {
    id: "soft-landing-set",
    name: "Soft Landing Complete Set",
    sub: "Whole set of 3 blind boxes",
    price: 19.99,
    stripe: "https://buy.stripe.com/6oU4gs3oH7Uk1cFczs4Rq00",
  },
};

const FREE_SHIPPING_MIN = 40;
const SHIPPING_FLAT = 4.99;

const SOFT_LANDING_DESC =
  "Our debut series introduces three collectible designs and one lucky variant, " +
  "each inspired by soft fur details and wing motifs. The wings reflect our vision: " +
  "dreams, creativity, and passion taking flight—while the idea of a " +
  "“soft landing” represents this first release: a gentle beginning to our " +
  "brand journey, created with care and intention.";

/* ---------- Helpers ---------- */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
const money = (n) => "$" + n.toFixed(2);
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 2400);
}

/* ---------- Cart state ---------- */

const CART_KEY = "cordi-cart";

function loadCart() {
  try {
    const raw = JSON.parse(localStorage.getItem(CART_KEY)) || {};
    const clean = {};
    for (const [id, qty] of Object.entries(raw)) {
      if (PRODUCTS[id] && Number.isInteger(qty) && qty > 0) clean[id] = Math.min(qty, 99);
    }
    return clean;
  } catch {
    return {};
  }
}

let cart = loadCart();

function saveCart() {
  localStorage.setItem(CART_KEY, JSON.stringify(cart));
}

function cartCount() {
  return Object.values(cart).reduce((a, b) => a + b, 0);
}

function cartSubtotal() {
  return Object.entries(cart).reduce((sum, [id, qty]) => sum + PRODUCTS[id].price * qty, 0);
}

function shippingFor(subtotal) {
  return subtotal === 0 || subtotal >= FREE_SHIPPING_MIN ? 0 : SHIPPING_FLAT;
}

function addToCart(id, qty) {
  cart[id] = Math.min((cart[id] || 0) + qty, 99);
  saveCart();
  renderCartUI();
  mascotReact("excited", "Yay! Coco tucked it into your cart ♡", 2800);
  toast("Added to cart");
}

function setQty(id, qty) {
  if (qty <= 0) delete cart[id];
  else cart[id] = Math.min(qty, 99);
  saveCart();
  renderCartUI();
}

function clearCart() {
  cart = {};
  saveCart();
  renderCartUI();
}

/* ---------- Cart drawer ---------- */

const drawer = $("#cart-drawer");
const backdrop = $("#drawer-backdrop");

function openCart() {
  renderCartUI();
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  backdrop.hidden = false;
}

function closeCart() {
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  backdrop.hidden = true;
}

function renderCartUI() {
  const count = cartCount();
  const badge = $("#cart-count");
  badge.hidden = count === 0;
  badge.textContent = count;

  const itemsEl = $("#cart-items");
  const footEl = $("#cart-foot");

  if (count === 0) {
    itemsEl.innerHTML =
      '<div class="cart-empty"><span class="big">🎀</span>' +
      "<p>Your cart is empty.</p><p>A blind box is waiting to meet you!</p></div>";
    footEl.innerHTML =
      '<a class="btn btn-pink btn-block" href="#/collections" data-close-cart>Browse Collections</a>';
    return;
  }

  itemsEl.innerHTML = Object.entries(cart)
    .map(([id, qty]) => {
      const p = PRODUCTS[id];
      return `
      <div class="cart-line">
        <div class="cart-thumb">?</div>
        <div>
          <h4>${esc(p.name)}</h4>
          <div class="cart-line-sub">${esc(p.sub)}</div>
          <div class="cart-line-qty">
            <button data-qty="${id}:-1" aria-label="Decrease quantity">−</button>
            <span>${qty}</span>
            <button data-qty="${id}:1" aria-label="Increase quantity">+</button>
          </div>
        </div>
        <div class="cart-line-right">
          <div class="price">${money(p.price * qty)}</div>
          <button class="cart-line-remove" data-remove="${id}">Remove</button>
        </div>
      </div>`;
    })
    .join("");

  const subtotal = cartSubtotal();
  const ship = shippingFor(subtotal);
  footEl.innerHTML = `
    <div class="cart-subtotal"><span>Subtotal</span><span>${money(subtotal)}</span></div>
    <p class="cart-shipnote">${
      ship === 0
        ? "You qualify for free shipping!"
        : `Add ${money(FREE_SHIPPING_MIN - subtotal)} more for free shipping.`
    }</p>
    <a class="btn btn-primary btn-block" href="#/checkout" data-close-cart>Check Out</a>`;
}

$("#cart-button").addEventListener("click", openCart);
$("#cart-close").addEventListener("click", closeCart);
backdrop.addEventListener("click", closeCart);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeCart(); });

drawer.addEventListener("click", (e) => {
  const t = e.target.closest("[data-qty], [data-remove], [data-close-cart]");
  if (!t) return;
  if (t.dataset.qty) {
    const [id, delta] = t.dataset.qty.split(":");
    setQty(id, (cart[id] || 0) + Number(delta));
  } else if (t.dataset.remove) {
    setQty(t.dataset.remove, 0);
  } else if (t.hasAttribute("data-close-cart")) {
    closeCart();
  }
});


/* ---------- Accounts and Coco Points (stored in this browser) ---------- */

const USERS_KEY = "cordi-users";
const SESSION_KEY = "cordi-session";
const POINTS_PER_DOLLAR = 10;   // earn 10 points per $1 spent
const POINTS_PER_DISCOUNT = 100; // every 100 points = $1 off

function loadUsers() {
  try { return JSON.parse(localStorage.getItem(USERS_KEY)) || {}; }
  catch { return {}; }
}

function saveUsers(users) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

async function hashPw(pw) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("cordi:" + pw));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function currentUser() {
  const email = localStorage.getItem(SESSION_KEY);
  if (!email) return null;
  const u = loadUsers()[email];
  return u ? { email, ...u } : null;
}

async function signUp(name, email, pw) {
  email = email.trim().toLowerCase();
  if (name.trim().length < 2) return "Please enter your name.";
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email)) return "Please enter a valid email.";
  if (pw.length < 6) return "Password needs at least 6 characters.";
  const users = loadUsers();
  if (users[email]) return "An account with that email already exists here.";
  users[email] = { name: name.trim(), pw: await hashPw(pw), points: 0, history: [] };
  saveUsers(users);
  localStorage.setItem(SESSION_KEY, email);
  return null;
}

async function signIn(email, pw) {
  email = email.trim().toLowerCase();
  const users = loadUsers();
  if (!users[email]) return "No account found with that email on this device.";
  if (users[email].pw !== await hashPw(pw)) return "That password doesn't match.";
  localStorage.setItem(SESSION_KEY, email);
  return null;
}

function signOut() {
  localStorage.removeItem(SESSION_KEY);
}

function adjustPoints(email, delta, reason) {
  const users = loadUsers();
  if (!users[email]) return;
  users[email].points = Math.max(0, users[email].points + delta);
  users[email].history.unshift({ delta, reason, date: new Date().toLocaleDateString() });
  users[email].history = users[email].history.slice(0, 30);
  saveUsers(users);
}

function renderAccountUI() {
  const btn = $("#account-button");
  if (btn) btn.classList.toggle("on", !!currentUser());
}

/* ---------- Icons ---------- */

const ICONS = {
  shield:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5c0 4.6-3 8.4-7 10-4-1.6-7-5.4-7-10V6l7-3z"/></svg>',
  eye:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-6.5 10-6.5S22 12 22 12s-3.5 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
  sparkle:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4L12 3z"/><path d="M19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16z"/></svg>',
  box:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8l-9-5-9 5v8l9 5 9-5V8z"/><path d="M3 8l9 5 9-5"/><path d="M12 13v8"/></svg>',
  lock:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="9" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>',
};

/* ---------- Pages ---------- */

function pageHome() {
  return `
  <div class="page">
    <section class="hero">
      <div class="hero-blob"></div>
      <div class="container">
        <p class="eyebrow">Photocard Accessories</p>
        <h1 class="display">Always<br><span class="accent">Facing</span><br>You.</h1>
        <p class="lead">Collectible blind box accessories designed to keep your photo cards front-facing—so your favorite idols and cherished moments are always on display.</p>
        <div class="hero-cta">
          <a class="btn btn-pink" href="#/collections">Shop Collections</a>
          <a class="btn btn-ghost" href="#/our-story">Our Story</a>
        </div>
      </div>
    </section>

    <section class="strip">
      <div class="container strip-grid">
        <div class="strip-item">
          <div class="strip-icon">${ICONS.shield}</div>
          <h3>Zero-Flip Promise</h3>
          <p>Two attachment rings keep your card front-facing through every commute, concert, and coffee run.</p>
        </div>
        <div class="strip-item">
          <div class="strip-icon">${ICONS.box}</div>
          <h3>Blind Box Joy</h3>
          <p>Every accessory arrives as a surprise. Three designs plus one lucky variant in every series.</p>
        </div>
        <div class="strip-item">
          <div class="strip-icon">${ICONS.sparkle}</div>
          <h3>Made With Care</h3>
          <p>Each piece is designed together by our whole team, for collectors who cherish their pulls.</p>
        </div>
      </div>
    </section>

    <section class="feature-banner">
      <div class="container">
        <div class="feature-card">
          <div>
            <p class="eyebrow">Now Available</p>
            <h2>Meet <span class="accent">Soft Landing</span>, our debut series</h2>
            <p>Three collectible designs plus one lucky variant, inspired by soft fur details and wing motifs. Which one will land with you?</p>
            <a class="btn btn-primary" href="#/collections/soft-landing">View the Series</a>
          </div>
          <div class="feature-visual"><div class="mystery-box">?</div></div>
        </div>
      </div>
    </section>
  </div>`;
}

function pageStory() {
  return `
  <div class="page">
    <section class="story-hero">
      <div class="container">
        <p class="eyebrow">How We Came To Be</p>
        <h1 class="display">Born From a <span class="accent">Collector's</span> Frustration</h1>
      </div>
    </section>

    <section class="story-body">
      <div class="container">
        <p>We made Cordi because we were honestly tired of one simple thing—turning our backpacks around and realizing our favorite photo card wasn’t even visible anymore. Half the time, it’s flipped, picture-hidden, facing the wrong way, and the moment you love just disappears. So we thought—why is it like this? Your favorite idol, that precious snapshot of a moment in your life, that one photo card you pulled—it deserves to be seen. So we created Cordi for you, someone who cares about their pulls, their biases, their cherished moments. Because if you're going to carry it with you, it should stay facing you—always.</p>
        <p>Every product goes through our zero-flip test—we simulate hours of movement to make sure your photo card stays exactly where it should.</p>
        <p>And the best part? All our photo card accessories come in blind box collections, turning each unboxing into a precious moment of surprise and joy in your day.</p>
      </div>
    </section>

    <section class="difference">
      <div class="container">
        <p class="eyebrow">The Cordi Lab Difference</p>
        <h2 class="display">Designed for You</h2>
        <div class="feature-list">
          <div class="feature-row">
            <div class="feature-icon">${ICONS.shield}</div>
            <div>
              <h3>Anti-Flip Engineering</h3>
              <p>We designed our holders with two attachment rings to keep your favorite photo card exactly where it belongs: front-facing and always visible to you and your world.</p>
            </div>
          </div>
          <div class="feature-row">
            <div class="feature-icon">${ICONS.eye}</div>
            <div>
              <h3>Collectible by Design</h3>
              <p>With four different designs in a single series including a "lucky" variant, Cordi merges our necessary anti-flip functionality with the exciting experience of blind box unboxing.</p>
            </div>
          </div>
          <div class="feature-row">
            <div class="feature-icon">${ICONS.sparkle}</div>
            <div>
              <h3>Handcrafted with Intention</h3>
              <p>Our designs are made together, with ideas from every member of our team. We create with the hope that each piece becomes something special—beautiful, collectable, and loved by Cordi's friends everywhere.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>`;
}

function pageCollections() {
  return `
  <div class="page">
    <section class="collections-hero">
      <div class="container">
        <p class="eyebrow">Our Series</p>
        <h1 class="display">Collections</h1>
        <p class="lead" style="margin-top:1.25rem">Every Cordi series is a small family of designs, sealed in blind boxes and waiting to surprise you.</p>
      </div>
    </section>
    <div class="container">
      <div class="series-grid">
        <a class="series-card" href="#/collections/soft-landing">
          <div class="series-cover">
            <span class="series-badge">Debut Series</span>
            <img src="img/gallery-1.jpeg" alt="Soft Landing blind box">
          </div>
          <div class="series-info">
            <h3>Soft Landing</h3>
            <p>Three collectible designs plus one lucky variant, inspired by soft fur details and wing motifs.</p>
            <div class="series-cta">Explore the series →</div>
          </div>
        </a>
      </div>
    </div>
  </div>`;
}

const GALLERY_IMAGES = [
  { src: "img/gallery-1.jpeg", alt: "Soft Landing blind box packaging with all four designs" },
  { src: "img/gallery-2.jpeg", alt: "Snow Angel design, pre-sale now" },
  { src: "img/gallery-3.jpeg", alt: "Silver Star design, pre-sale now" },
  { src: "img/gallery-4.jpeg", alt: "Midnight Flight design, pre-sale now" },
  { src: "img/gallery-5.png", alt: "Teddy Wing hidden edition, pre-sale now" },
];
const GALLERY_SLIDES = GALLERY_IMAGES.length;

function pageProduct() {
  const slides = GALLERY_IMAGES.map((img, i) => `
    <div class="gallery-slide" role="group" aria-label="Photo ${i + 1} of ${GALLERY_SLIDES}">
      <img src="${img.src}" alt="${esc(img.alt)}" loading="${i === 0 ? "eager" : "lazy"}">
    </div>`).join("");

  const dots = Array.from({ length: GALLERY_SLIDES }, (_, i) =>
    `<button data-dot="${i}" class="${i === 0 ? "active" : ""}" aria-label="Go to photo ${i + 1}"></button>`).join("");

  return `
  <div class="page product-page">
    <div class="container">
      <p class="breadcrumb"><a href="#/collections">Collections</a> / Soft Landing</p>
      <div class="product-layout">
        <div class="gallery">
          <div class="gallery-track" id="gallery-track">${slides}</div>
          <button class="gallery-nav prev" id="gallery-prev" aria-label="Previous photo">←</button>
          <button class="gallery-nav next" id="gallery-next" aria-label="Next photo">→</button>
          <div class="gallery-dots" id="gallery-dots">${dots}</div>
          <p class="gallery-note">Concept renders. Production photos coming soon!</p>
        </div>

        <div class="product-info">
          <p class="eyebrow">Debut Series</p>
          <h1>Soft <span class="accent">Landing</span></h1>
          <p class="product-tagline">A blind box photo card holder series. Three designs, one lucky variant, endless anti-flip peace of mind.</p>

          <div class="option-group">
            <span>Choose your box</span>
            <div class="option-cards" id="option-cards">
              <button class="option-card selected" data-option="soft-landing-single" aria-pressed="true">
                <span><span class="opt-name">Single Blind Box</span>
                <span class="opt-sub">One surprise design</span></span>
                <span><span class="opt-price">${money(PRODUCTS["soft-landing-single"].price)}</span></span>
              </button>
              <button class="option-card" data-option="soft-landing-set" aria-pressed="false">
                <span><span class="opt-name">Whole Set</span>
                <span class="opt-sub">3 blind boxes, one of each design</span></span>
                <span><span class="opt-price">${money(PRODUCTS["soft-landing-set"].price)}</span>
                <span class="opt-save">Save ${money(PRODUCTS["soft-landing-single"].price * 3 - PRODUCTS["soft-landing-set"].price)}</span></span>
              </button>
            </div>
          </div>

          <div class="qty-row">
            <div class="qty-stepper">
              <button id="qty-minus" aria-label="Decrease quantity">−</button>
              <span id="qty-value">1</span>
              <button id="qty-plus" aria-label="Increase quantity">+</button>
            </div>
            <button class="btn btn-pink" id="add-to-cart">Add to Cart · <span id="atc-price">${money(PRODUCTS["soft-landing-single"].price)}</span></button>
          </div>

          <div class="product-desc">
            <h2>About this series</h2>
            <p>${SOFT_LANDING_DESC}</p>
          </div>
        </div>
      </div>
    </div>
  </div>`;
}

function bindProductPage() {
  let selected = "soft-landing-single";
  let qty = 1;

  const priceEl = $("#atc-price");
  const qtyEl = $("#qty-value");

  function refresh() {
    qtyEl.textContent = qty;
    priceEl.textContent = money(PRODUCTS[selected].price * qty);
  }

  $$("#option-cards .option-card").forEach((card) => {
    card.addEventListener("click", () => {
      selected = card.dataset.option;
      $$("#option-cards .option-card").forEach((c) => {
        const on = c === card;
        c.classList.toggle("selected", on);
        c.setAttribute("aria-pressed", String(on));
      });
      refresh();
    });
  });

  $("#qty-minus").addEventListener("click", () => { qty = Math.max(1, qty - 1); refresh(); });
  $("#qty-plus").addEventListener("click", () => { qty = Math.min(99, qty + 1); refresh(); });
  $("#add-to-cart").addEventListener("click", () => { addToCart(selected, qty); openCart(); });

  // Gallery
  const track = $("#gallery-track");
  const dots = $$("#gallery-dots button");
  const slideStep = () =>
    track.children.length > 1
      ? track.children[1].offsetLeft - track.children[0].offsetLeft
      : track.clientWidth;
  const slideTo = (i) => {
    const idx = Math.max(0, Math.min(GALLERY_SLIDES - 1, i));
    track.scrollTo({ left: idx * slideStep(), behavior: "smooth" });
  };
  const currentSlide = () => Math.round(track.scrollLeft / slideStep());

  $("#gallery-prev").addEventListener("click", () => slideTo(currentSlide() - 1));
  $("#gallery-next").addEventListener("click", () => slideTo(currentSlide() + 1));
  dots.forEach((d) => d.addEventListener("click", () => slideTo(Number(d.dataset.dot))));

  track.addEventListener("scroll", () => {
    const idx = currentSlide();
    dots.forEach((d, i) => d.classList.toggle("active", i === idx));
  }, { passive: true });
}


function pageAccount() {
  const user = currentUser();
  if (!user) {
    return `
  <div class="page account-page">
    <div class="container">
      <p class="eyebrow">Your Burrow</p>
      <h1 class="display">Join the <span class="accent">fluffle</span></h1>
      <p class="lead" style="margin-top:1rem">Create an account to collect Coco Points. Earn ${POINTS_PER_DOLLAR} points for every $1 you spend, and every ${POINTS_PER_DISCOUNT} points takes $1 off a future order.</p>
      <div class="auth-card">
        <div class="auth-tabs">
          <button class="active" data-tab="signup">Create account</button>
          <button data-tab="signin">Sign in</button>
        </div>
        <form id="auth-form" novalidate>
          <div class="field" id="auth-name-field">
            <label for="a-name">Name</label>
            <input id="a-name" autocomplete="name" placeholder="Coco Bunny">
          </div>
          <div class="field">
            <label for="a-email">Email</label>
            <input id="a-email" type="email" autocomplete="email" placeholder="you@example.com">
          </div>
          <div class="field">
            <label for="a-pw">Password</label>
            <input id="a-pw" type="password" autocomplete="new-password" placeholder="At least 6 characters">
          </div>
          <p class="auth-error" id="auth-error" hidden></p>
          <button type="submit" class="btn btn-pink btn-block" id="auth-submit">Create account</button>
        </form>
        <p class="auth-note">Demo accounts live in this browser only. Real cross device accounts arrive with the production launch.</p>
      </div>
    </div>
  </div>`;
  }

  const dollars = Math.floor(user.points / POINTS_PER_DISCOUNT);
  const history = (user.history || []).map((h) => `
    <li><span>${esc(h.reason)}</span><span class="${h.delta >= 0 ? "gain" : "spend"}">${h.delta >= 0 ? "+" : ""}${h.delta} pts</span><span class="muted">${esc(h.date)}</span></li>`).join("");

  return `
  <div class="page account-page">
    <div class="container">
      <p class="eyebrow">Your Burrow</p>
      <h1 class="display">Hi, <span class="accent">${esc(user.name)}</span></h1>
      <div class="points-card">
        <div>
          <div class="points-big">${user.points}</div>
          <div class="points-label">Coco Points</div>
        </div>
        <p class="points-worth">${dollars > 0 ? `Worth $${dollars} off your next order ♡` : `Earn ${POINTS_PER_DOLLAR} points per $1 spent. ${POINTS_PER_DISCOUNT} points = $1 off.`}</p>
      </div>
      <div class="account-columns">
        <div>
          <h2 class="account-h2">How points work</h2>
          <ul class="points-rules">
            <li>Earn ${POINTS_PER_DOLLAR} points for every $1 you spend</li>
            <li>Every ${POINTS_PER_DISCOUNT} points = $1 off at checkout</li>
            <li>Points apply automatically when you choose to redeem</li>
          </ul>
        </div>
        <div>
          <h2 class="account-h2">Points history</h2>
          ${history ? `<ul class="points-history">${history}</ul>` : `<p class="muted-note">No points yet. Your first blind box fixes that ♡</p>`}
        </div>
      </div>
      <button class="btn btn-ghost" id="signout-btn" style="margin-top:2rem">Sign out</button>
    </div>
  </div>`;
}

function bindAccountPage() {
  const so = $("#signout-btn");
  if (so) {
    so.addEventListener("click", () => { signOut(); renderAccountUI(); render(); });
    return;
  }
  let mode = "signup";
  const form = $("#auth-form");
  const err = $("#auth-error");
  $$(".auth-tabs button").forEach((b) => b.addEventListener("click", () => {
    mode = b.dataset.tab;
    $$(".auth-tabs button").forEach((x) => x.classList.toggle("active", x === b));
    $("#auth-name-field").style.display = mode === "signup" ? "" : "none";
    $("#auth-submit").textContent = mode === "signup" ? "Create account" : "Sign in";
    err.hidden = true;
  }));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#a-name").value, email = $("#a-email").value, pw = $("#a-pw").value;
    const problem = mode === "signup" ? await signUp(name, email, pw) : await signIn(email, pw);
    if (problem) {
      err.textContent = problem;
      err.hidden = false;
      return;
    }
    renderAccountUI();
    mascotReact("excited", mode === "signup" ? "Welcome to the fluffle! ♡" : "Welcome back! Coco missed you ♡", 2800);
    render();
  });
}

/* ---------- Checkout ---------- */

function pageCheckout() {
  if (cartCount() === 0) {
    return `
    <div class="page confirm-page">
      <div class="container">
        <h1 class="display">Your cart is <span class="accent">empty</span></h1>
        <p class="lead">Add a blind box first, then come back to check out.</p>
        <a class="btn btn-pink" href="#/collections">Browse Collections</a>
      </div>
    </div>`;
  }

  const subtotal = cartSubtotal();
  const ship = shippingFor(subtotal);
  const total = subtotal + ship;

  const lines = Object.entries(cart).map(([id, qty]) => {
    const p = PRODUCTS[id];
    return `<div class="summary-line"><span class="muted">${esc(p.name)} × ${qty}</span><span>${money(p.price * qty)}</span></div>`;
  }).join("");

  return `
  <div class="page checkout-page">
    <div class="container">
      <h1 class="display">Checkout</h1>
      <p class="checkout-sub">You're almost there. Cordi is cheering you on!</p>
      <div class="checkout-layout">
        <form class="checkout-form" id="checkout-form" novalidate>
          <fieldset>
            <legend>Contact &amp; Shipping</legend>
            <div class="field" data-field="name">
              <label for="f-name">Full name</label>
              <input id="f-name" name="name" autocomplete="name" placeholder="Cordi Bunny">
              <p class="error">Please enter your name.</p>
            </div>
            <div class="field" data-field="email">
              <label for="f-email">Email</label>
              <input id="f-email" name="email" type="email" autocomplete="email" placeholder="you@example.com">
              <p class="error">Please enter a valid email address.</p>
            </div>
            <div class="field" data-field="address">
              <label for="f-address">Street address</label>
              <input id="f-address" name="address" autocomplete="street-address" placeholder="123 Cloud Lane">
              <p class="error">Please enter your street address.</p>
            </div>
            <div class="field-row">
              <div class="field" data-field="city">
                <label for="f-city">City</label>
                <input id="f-city" name="city" autocomplete="address-level2" placeholder="Seattle">
                <p class="error">Please enter your city.</p>
              </div>
              <div class="field" data-field="zip">
                <label for="f-zip">ZIP / Postal code</label>
                <input id="f-zip" name="zip" autocomplete="postal-code" placeholder="98101">
                <p class="error">Please enter a valid postal code.</p>
              </div>
            </div>
          </fieldset>

          <fieldset>
            <legend>Payment</legend>
            ${(() => {
              const lines = Object.entries(cart).filter(([id]) => PRODUCTS[id].stripe);
              if (!lines.length) return "";
              const testMode = lines.some(([id]) => PRODUCTS[id].stripe.includes("/test_"));
              return `<div class="stripe-pay">
                ${lines.map(([id, qty]) => {
                  const p = PRODUCTS[id];
                  return `<a class="btn btn-primary btn-block" href="${p.stripe}" target="_blank" rel="noopener">Pay for ${esc(p.name)}${qty > 1 ? " × " + qty : ""} with Stripe</a>`;
                }).join("")}
                <p class="stripe-note">Secure checkout by Stripe: card, Apple Pay, Google Pay. Set your quantity on the Stripe page.${testMode ? " <b>Test mode is on, so no real charges yet.</b>" : ""}</p>
                <div class="or-divider"><span>or try the demo checkout</span></div>
              </div>`;
            })()}
            <div class="field" data-field="cardName">
              <label for="f-card-name">Name on card</label>
              <input id="f-card-name" name="cardName" autocomplete="cc-name" placeholder="Cordi Bunny">
              <p class="error">Please enter the name on your card.</p>
            </div>
            <div class="field" data-field="cardNumber">
              <label for="f-card-number">Card number</label>
              <div class="card-input-wrap">
                <input id="f-card-number" name="cardNumber" inputmode="numeric" autocomplete="cc-number" placeholder="1234 5678 9012 3456" maxlength="19">
                <span class="card-brand" id="card-brand"></span>
              </div>
              <p class="error">Please enter a valid card number.</p>
            </div>
            <div class="field-row">
              <div class="field" data-field="expiry">
                <label for="f-expiry">Expiry (MM/YY)</label>
                <input id="f-expiry" name="expiry" inputmode="numeric" autocomplete="cc-exp" placeholder="08/28" maxlength="5">
                <p class="error">Please enter a valid future expiry date.</p>
              </div>
              <div class="field" data-field="cvc">
                <label for="f-cvc">Security code</label>
                <input id="f-cvc" name="cvc" inputmode="numeric" autocomplete="cc-csc" placeholder="123" maxlength="4">
                <p class="error">Please enter the 3 or 4 digit code.</p>
              </div>
            </div>
            ${(() => {
              const u = currentUser();
              if (!u) return '<p class="points-nudge">Psst: <a href="#/account">create an account</a> to earn Coco Points on this order.</p>';
              const maxD = Math.min(Math.floor(u.points / POINTS_PER_DISCOUNT), Math.floor(subtotal));
              if (maxD < 1) return `<p class="points-nudge">You'll earn Coco Points on this order ♡</p>`;
              return `<label class="points-redeem"><input type="checkbox" id="use-points" data-discount="${maxD}"> Use ${maxD * POINTS_PER_DISCOUNT} of my ${u.points} Coco Points (−${money(maxD)})</label>`;
            })()}
            <p class="secure-note">${ICONS.lock} Your details are encrypted and secure. This demo store does not charge real cards.</p>
            <button type="submit" class="btn btn-primary btn-block" id="pay-button">Pay ${money(total)}</button>
          </fieldset>
        </form>

        <aside class="order-summary">
          <h2>Order Summary</h2>
          ${lines}
          <div class="summary-line"><span class="muted">Shipping</span><span>${ship === 0 ? "Free" : money(ship)}</span></div>
          <div class="summary-line" id="sum-discount" hidden><span class="muted">Coco Points</span><span id="sum-discount-val"></span></div>
          <div class="summary-total"><span>Total</span><span id="sum-total">${money(total)}</span></div>
        </aside>
      </div>
    </div>
  </div>`;
}

function luhnValid(digits) {
  let sum = 0;
  let dbl = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = Number(digits[i]);
    if (dbl) { d *= 2; if (d > 9) d -= 9; }
    sum += d;
    dbl = !dbl;
  }
  return sum % 10 === 0;
}

function cardBrand(digits) {
  if (/^4/.test(digits)) return "VISA";
  if (/^(5[1-5]|2[2-7])/.test(digits)) return "MASTERCARD";
  if (/^3[47]/.test(digits)) return "AMEX";
  if (/^6(011|5)/.test(digits)) return "DISCOVER";
  return "";
}

function expiryValid(value) {
  const m = value.match(/^(\d{2})\/(\d{2})$/);
  if (!m) return false;
  const month = Number(m[1]);
  if (month < 1 || month > 12) return false;
  const year = 2000 + Number(m[2]);
  const now = new Date();
  return year > now.getFullYear() || (year === now.getFullYear() && month >= now.getMonth() + 1);
}

const VALIDATORS = {
  name: (v) => v.trim().length >= 2,
  email: (v) => /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(v.trim()),
  address: (v) => v.trim().length >= 4,
  city: (v) => v.trim().length >= 2,
  zip: (v) => /^[A-Za-z0-9][A-Za-z0-9\s-]{2,9}$/.test(v.trim()),
  cardName: (v) => v.trim().length >= 2,
  cardNumber: (v) => {
    const digits = v.replace(/\D/g, "");
    return digits.length >= 15 && digits.length <= 16 && luhnValid(digits);
  },
  expiry: expiryValid,
  cvc: (v) => /^\d{3,4}$/.test(v.trim()),
};

function bindCheckoutPage() {
  const form = $("#checkout-form");
  if (!form) return;

  const cardInput = $("#f-card-number");
  const brandEl = $("#card-brand");
  cardInput.addEventListener("input", () => {
    const digits = cardInput.value.replace(/\D/g, "").slice(0, 16);
    cardInput.value = digits.replace(/(\d{4})(?=\d)/g, "$1 ");
    brandEl.textContent = cardBrand(digits);
  });

  const expiryInput = $("#f-expiry");
  expiryInput.addEventListener("input", () => {
    let digits = expiryInput.value.replace(/\D/g, "").slice(0, 4);
    if (digits.length >= 3) digits = digits.slice(0, 2) + "/" + digits.slice(2);
    expiryInput.value = digits;
  });

  $("#f-cvc").addEventListener("input", (e) => {
    e.target.value = e.target.value.replace(/\D/g, "").slice(0, 4);
  });

  function validateField(wrap) {
    const name = wrap.dataset.field;
    const input = wrap.querySelector("input");
    const ok = VALIDATORS[name](input.value);
    wrap.classList.toggle("show-error", !ok);
    input.classList.toggle("invalid", !ok);
    return ok;
  }

  $$(".field", form).forEach((wrap) => {
    wrap.querySelector("input").addEventListener("blur", () => validateField(wrap));
    wrap.querySelector("input").addEventListener("input", () => {
      if (wrap.classList.contains("show-error")) validateField(wrap);
    });
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const wraps = $$(".field", form);
    let firstBad = null;
    for (const wrap of wraps) {
      if (!validateField(wrap) && !firstBad) firstBad = wrap;
    }
    if (firstBad) {
      firstBad.querySelector("input").focus();
      mascotReact("wink", "Oops! Coco spotted a field to fix.", 3000);
      return;
    }

    const payBtn = $("#pay-button");
    payBtn.disabled = true;
    payBtn.textContent = "Processing…";

    // Simulated payment processing
    setTimeout(() => {
      const usePts = $("#use-points");
      const discount = usePts && usePts.checked ? Number(usePts.dataset.discount) : 0;
      const paid = Math.max(0, cartSubtotal() + shippingFor(cartSubtotal()) - discount);
      const user = currentUser();
      let earned = 0;
      if (user) {
        if (discount > 0) adjustPoints(user.email, -discount * POINTS_PER_DISCOUNT, "Redeemed at checkout");
        earned = Math.floor(paid * POINTS_PER_DOLLAR);
        if (earned > 0) adjustPoints(user.email, earned, "Order " + "CL-" + Date.now().toString(36).toUpperCase().slice(-5));
      }
      lastOrder = {
        number: "CL-" + Date.now().toString(36).toUpperCase(),
        total: paid,
        email: $("#f-email").value.trim(),
        pointsEarned: earned,
      };
      clearCart();
      location.hash = "#/confirmation";
    }, 1200);
  });

  const usePts = $("#use-points");
  if (usePts) {
    usePts.addEventListener("change", () => {
      const d = usePts.checked ? Number(usePts.dataset.discount) : 0;
      const total = Math.max(0, cartSubtotal() + shippingFor(cartSubtotal()) - d);
      $("#sum-discount").hidden = d === 0;
      $("#sum-discount-val").textContent = "−" + money(d);
      $("#sum-total").textContent = money(total);
      $("#pay-button").textContent = "Pay " + money(total);
    });
  }
}

/* ---------- Confirmation ---------- */

let lastOrder = null;

function pageConfirmation() {
  if (!lastOrder) {
    return `
    <div class="page confirm-page">
      <div class="container">
        <h1 class="display">Nothing here <span class="accent">yet</span></h1>
        <p class="lead">Your next order confirmation will land on this page.</p>
        <a class="btn btn-pink" href="#/collections">Browse Collections</a>
      </div>
    </div>`;
  }
  return `
  <div class="page confirm-page">
    <div class="container">
      <p class="eyebrow">Order Confirmed</p>
      <h1 class="display">Thank you! Your surprise is <span class="accent">on its way.</span></h1>
      <p class="lead">We’ve emailed a receipt to ${esc(lastOrder.email)}. Your blind box will ship soon, and we can’t wait for you to meet whoever’s inside.</p>
      <div class="order-number">Order ${esc(lastOrder.number)} · ${money(lastOrder.total)}</div>
      ${lastOrder.pointsEarned ? `<p class="points-earned">+${lastOrder.pointsEarned} Coco Points earned ♡ <a href="#/account">See your balance</a></p>` : `<p class="points-earned muted-note"><a href="#/account">Create an account</a> to earn Coco Points next time!</p>`}
      <div><a class="btn btn-primary" href="#/">Back to Home</a></div>
    </div>
  </div>`;
}

/* ---------- Mascot ---------- */

const BUNNY = {
  base(parts) {
    const p = Object.assign({ eyes: "open", extra: "", arms: "down", ears: "" }, parts);

    const eyeOpen = (cx) =>
      `<g class="eye"><circle cx="${cx}" cy="62" r="5.2" fill="#3a2e26"/>` +
      `<circle class="pupil-glint" cx="${cx + 1.6}" cy="60.2" r="1.6" fill="#fff"/></g>`;
    const eyeWink = (cx) =>
      `<path d="M${cx - 5} 62 q5 -5 10 0" stroke="#3a2e26" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
    const eyeSleep = (cx) =>
      `<path d="M${cx - 5} 63 q5 4 10 0" stroke="#3a2e26" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
    const eyeHappy = (cx) =>
      `<path d="M${cx - 5} 63 q5 -6 10 0" stroke="#3a2e26" stroke-width="2.6" fill="none" stroke-linecap="round"/>`;

    let leftEye, rightEye;
    if (p.eyes === "open") { leftEye = eyeOpen(47); rightEye = eyeOpen(73); }
    else if (p.eyes === "wink") { leftEye = eyeOpen(47); rightEye = eyeWink(73); }
    else if (p.eyes === "sleep") { leftEye = eyeSleep(47); rightEye = eyeSleep(73); }
    else { leftEye = eyeHappy(47); rightEye = eyeHappy(73); }

    const armsDown =
      `<circle cx="34" cy="88" r="9" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>` +
      `<circle cx="86" cy="88" r="9" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>`;
    const armsUp =
      `<circle cx="26" cy="62" r="9" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>` +
      `<circle cx="94" cy="62" r="9" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>`;
    const armsFront =
      `<circle cx="44" cy="94" r="9" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>` +
      `<circle cx="76" cy="94" r="9" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>`;

    const arms = p.arms === "up" ? armsUp : p.arms === "front" ? armsFront : armsDown;

    return `
    <svg viewBox="0 0 120 130" xmlns="http://www.w3.org/2000/svg">
      <g class="pose">
        ${p.extra.includes("BEHIND:") ? p.extra.split("BEHIND:")[1] : ""}
        <g class="ears">
          <ellipse cx="44" cy="26" rx="10" ry="21" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2" transform="rotate(-8 44 26)"/>
          <ellipse cx="44" cy="29" rx="5" ry="13" fill="#f6d3d7" transform="rotate(-8 44 29)"/>
          <ellipse cx="76" cy="26" rx="10" ry="21" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2" transform="rotate(8 76 26)"/>
          <ellipse cx="76" cy="29" rx="5" ry="13" fill="#f6d3d7" transform="rotate(8 76 29)"/>
          <path d="M84 14 c2.5 -3.5 7.5 -1.5 7.5 2 c0 3 -4 5.5 -7.5 7.5 c-3.5 -2 -7.5 -4.5 -7.5 -7.5 c0 -3.5 5 -5.5 7.5 -2z" fill="#ef9aa6"/>
        </g>
        <ellipse cx="60" cy="96" rx="32" ry="24" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>
        <circle cx="60" cy="60" r="30" fill="#fdf6ee" stroke="#dcc9b6" stroke-width="2"/>
        <circle cx="40" cy="70" r="6" fill="#f6bfc6" opacity="0.85"/>
        <circle cx="80" cy="70" r="6" fill="#f6bfc6" opacity="0.85"/>
        <g class="eyes-group">${leftEye}${rightEye}</g>
        <path d="M57 71 q3 2.6 6 0" stroke="#3a2e26" stroke-width="2" fill="none" stroke-linecap="round"/>
        ${arms}
        ${p.extra.includes("BEHIND:") ? "" : p.extra}
      </g>
    </svg>`;
  },

  poses: {
    sit() { return BUNNY.base({}); },
    wink() { return BUNNY.base({ eyes: "wink" }); },
    excited() {
      return BUNNY.base({
        eyes: "happy",
        arms: "up",
        extra:
          `<path d="M14 40 l4 -8 M22 34 l1 -8 M100 34 l-1 -8 M106 40 l-4 -8" stroke="#e8a0aa" stroke-width="2.4" stroke-linecap="round"/>`,
      });
    },
    heart() {
      return BUNNY.base({
        eyes: "happy",
        arms: "front",
        extra:
          `<path d="M60 84 c6 -9 19 -4 19 5 c0 8 -10 14 -19 19 c-9 -5 -19 -11 -19 -19 c0 -9 13 -14 19 -5z" fill="#ef9aa6"/>`,
      });
    },
    star() {
      return BUNNY.base({
        eyes: "open",
        arms: "front",
        extra:
          `<path d="M60 80 l4.6 9.3 10.4 1.5 -7.5 7.3 1.8 10.3 -9.3 -4.9 -9.3 4.9 1.8 -10.3 -7.5 -7.3 10.4 -1.5z" fill="#f7d67c" stroke="#e8bd55" stroke-width="1.5"/>`,
      });
    },
    reading() {
      return BUNNY.base({
        eyes: "sleep",
        arms: "front",
        extra:
          `<g><path d="M38 86 q22 -8 44 0 l0 22 q-22 -8 -44 0z" fill="#f2aab4" stroke="#dd8f9b" stroke-width="1.5"/>` +
          `<path d="M60 82.5 l0 22" stroke="#dd8f9b" stroke-width="1.5"/></g>`,
      });
    },
    sleep() {
      return BUNNY.base({
        eyes: "sleep",
        extra:
          `<text x="92" y="38" font-family="Georgia, serif" font-size="13" fill="#b9a893" font-style="italic">z</text>` +
          `<text x="100" y="28" font-family="Georgia, serif" font-size="16" fill="#b9a893" font-style="italic">z</text>`,
      });
    },
    parachute() {
      return BUNNY.base({
        eyes: "open",
        extra:
          "BEHIND:" +
          `<g><path d="M18 30 a42 34 0 0 1 84 0 q-10 8 -21 0 q-10 8 -21 0 q-10 8 -21 0 q-10 8 -21 0z" fill="#f2aab4" stroke="#dd8f9b" stroke-width="1.5" opacity="0.95"/>` +
          `<path d="M22 32 L38 58 M60 34 L60 52 M98 32 L82 58" stroke="#c9b6a2" stroke-width="1.4"/></g>`,
      });
    },
  },
};

const mascotEl = $("#mascot");
const mascotStage = $("#mascot-stage");
const mascotBubble = $("#mascot-bubble");

let currentPose = "";
let poseLock = 0;

function setMascotPose(pose, force) {
  if (!force && Date.now() < poseLock) return;
  if (pose === currentPose && !force) return;
  currentPose = pose;
  mascotStage.innerHTML = (BUNNY.poses[pose] || BUNNY.poses.sit)();
  const g = mascotStage.querySelector(".pose");
  if (g) {
    g.classList.add("hop");
    setTimeout(() => g.classList.remove("hop"), 600);
  }
}

function sayBubble(text, ms) {
  if (typeof chatOpen !== "undefined" && chatOpen) return;
  mascotBubble.textContent = text;
  mascotBubble.hidden = false;
  clearTimeout(sayBubble._t);
  sayBubble._t = setTimeout(() => { mascotBubble.hidden = true; }, ms || 3200);
}

function mascotReact(pose, text, ms) {
  poseLock = Date.now() + (ms || 2500);
  setMascotPose(pose, true);
  if (text) sayBubble(text, ms);
  setTimeout(() => {
    poseLock = 0;
    setMascotPose(ROUTE_POSE[currentRouteKey] || "sit", true);
  }, ms || 2500);
}

/* Eyes follow the cursor for the "watching you" illusion */
let mouseX = innerWidth / 2;
let mouseY = innerHeight / 2;
let eyeRaf = null;

document.addEventListener("mousemove", (e) => {
  mouseX = e.clientX;
  mouseY = e.clientY;
  wakeMascot();
  if (!eyeRaf) eyeRaf = requestAnimationFrame(updateEyes);
});

function updateEyes() {
  eyeRaf = null;
  const eyes = mascotStage.querySelector(".eyes-group");
  if (!eyes) return;
  const rect = mascotStage.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height * 0.45;
  const dx = mouseX - cx;
  const dy = mouseY - cy;
  const dist = Math.hypot(dx, dy) || 1;
  const r = Math.min(2.6, dist / 40);
  eyes.style.transform = `translate(${(dx / dist) * r}px, ${(dy / dist) * r}px)`;
}

/* Ears perk when scrolling; sleeps when idle */
let idleTimer = null;
let asleep = false;

function wakeMascot() {
  if (asleep) {
    asleep = false;
    setMascotPose(ROUTE_POSE[currentRouteKey] || "sit", true);
    sayBubble("Oh! Coco's awake! Did you miss me?", 2600);
  }
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    asleep = true;
    if (Date.now() >= poseLock) {
      setMascotPose("sleep", true);
      sayBubble("Coco's taking a tiny nap… zzz", 3000);
    }
  }, 45000);
}

["scroll", "keydown", "touchstart"].forEach((ev) =>
  addEventListener(ev, wakeMascot, { passive: true })
);

/* ---------- Coco chat ---------- */

const CHAT_TOPICS = [
  { keys: ["hi", "hello", "hey", "sup", "yo "], a: "Hi hi! I'm Coco ♡ Ask me about blind boxes, prices, shipping, or the lucky variant!" },
  { keys: ["help", "confused", "question", "how does"], a: "Happy to help! You can ask me about blind boxes, prices, shipping, the lucky variant, or how ordering works ♡" },
  { keys: ["lucky", "variant", "rare", "secret", "chase"], a: "Ooh, the lucky variant! Soft Landing has three regular designs plus one lucky variant, so four designs in total. Any box could be the lucky one…" },
  { keys: ["blind box", "blindbox", "blind-box", "surprise", "mystery"], a: "A blind box is a sealed little mystery! You won't know which of the four designs is inside until you open it. That's the best part ♡" },
  { keys: ["set", "all of them", "every design", "duplicates"], a: "The Whole Set is 3 blind boxes, one of each design, for $19.99. No duplicates, and it saves you $3.98 versus singles!" },
  { keys: ["ship", "shipping cost", "deliver", "mail", "arrive"], a: "Shipping is a flat $4.99, and free once your order reaches $40!" },
  { keys: ["price", "cost", "how much", "expensive"], a: "A single blind box is $7.99, and the whole set of 3 is $19.99 ♡" },
  { keys: ["flip", "anti", "ring", "facing", "backwards"], a: "Every holder has two attachment rings so your photo card stays front-facing, always. Zero flips, bunny promise!" },
  { keys: ["fit", "size", "photocard", "photo card", "dimension"], a: "Cordi holders fit standard photo cards, the 55 by 85 mm kind you pull from albums ♡" },
  { keys: ["return", "refund", "cancel", "real card", "charge"], a: "Little secret: this is a demo storefront, so payments are simulated. No real charges, promise!" },
  { keys: ["order", "track", "status", "receipt"], a: "Demo orders don't really ship (yet!), so your order number is just a keepsake for now ♡" },
  { keys: ["buy", "checkout", "cart", "pay", "purchase"], a: "Open Collections, pick Soft Landing, choose a single box or the whole set, then tap the cart up top to check out. I'll cheer the whole way!" },
  { keys: ["story", "about", "cordi", "brand", "who made"], a: "Cordi Lab was born from a collector's frustration with flipped photo cards. The whole story is on the Our Story page ♡" },
  { keys: ["photo", "picture", "image", "prototype"], a: "Product photos are coming soon! We're waiting on the box and prototype, and I can't wait to show you." },
  { keys: ["coco", "cute", "bunny", "name"], a: "Hehe, that's me! I'm Coco, the Cordi Lab bunny ♡" },
];

CHAT_TOPICS.push(
  { keys: ["point", "loyalty", "reward", "account", "sign up", "signup"], a: "Coco Points! Earn 10 points for every $1 you spend, and every 100 points takes $1 off a future order. Make an account on the Account page (the little bunny-person icon up top) to start collecting ♡" }
);

const CHAT_FALLBACK =
  "Hmm, Coco's not sure about that one! Try asking about blind boxes, prices, shipping, or the lucky variant ♡";

const CHAT_CHIPS = ["What's a blind box?", "Shipping", "The lucky variant", "How do I order?"];

function cocoAnswer(question) {
  const t = question.toLowerCase();
  let best = null;
  let bestScore = 0;
  for (const topic of CHAT_TOPICS) {
    let score = 0;
    for (const k of topic.keys) {
      // Keys match at word starts only, so "hi" can't hide inside "shipping"
      const re = new RegExp("\\b" + k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      if (re.test(t)) score++;
    }
    if (score > bestScore) { bestScore = score; best = topic; }
  }
  return best ? best.a : CHAT_FALLBACK;
}

const chatEl = $("#coco-chat");
const chatMessages = $("#coco-chat-messages");
const chatChips = $("#coco-chat-chips");
const chatForm = $("#coco-chat-form");
const chatInput = $("#coco-chat-input");

let chatOpen = false;
let chatGreeted = false;

function pushChatMsg(text, who) {
  const el = document.createElement("div");
  el.className = "chat-msg " + who;
  el.textContent = text;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function cocoReplyTo(question) {
  const typing = pushChatMsg("Coco is typing…", "coco typing");
  setTimeout(() => {
    typing.classList.remove("typing");
    typing.textContent = cocoAnswer(question);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 550 + Math.random() * 500);
}

function askCoco(question) {
  pushChatMsg(question, "user");
  cocoReplyTo(question);
}

function openChat() {
  chatOpen = true;
  chatEl.hidden = false;
  mascotBubble.hidden = true;
  if (!chatGreeted) {
    chatGreeted = true;
    pushChatMsg("Hi! I'm Coco ♡ Confused about anything, or just curious? Ask away, or tap a question below!", "coco");
    chatChips.innerHTML = CHAT_CHIPS
      .map((c) => `<button type="button">${esc(c)}</button>`)
      .join("");
  }
  chatInput.focus();
  setMascotPose("excited", true);
  setTimeout(() => { if (chatOpen) setMascotPose(ROUTE_POSE[currentRouteKey] || "sit", true); }, 900);
}

function closeChat() {
  chatOpen = false;
  chatEl.hidden = true;
}

$("#mascot-stage").addEventListener("click", () => {
  wakeMascot();
  chatOpen ? closeChat() : openChat();
});
$("#coco-chat-close").addEventListener("click", closeChat);

chatChips.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) askCoco(btn.textContent);
});

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = chatInput.value.trim();
  if (!q) return;
  chatInput.value = "";
  askCoco(q);
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeChat();
});

/* ---------- Router ---------- */

const ROUTES = {
  home: { render: pageHome, nav: "home" },
  "our-story": { render: pageStory, nav: "our-story" },
  collections: { render: pageCollections, nav: "collections" },
  product: { render: pageProduct, nav: "collections", bind: bindProductPage },
  checkout: { render: pageCheckout, nav: null, bind: bindCheckoutPage },
  account: { render: pageAccount, nav: null, bind: bindAccountPage },
  confirmation: { render: pageConfirmation, nav: null },
};

const ROUTE_POSE = {
  home: "parachute",
  "our-story": "reading",
  collections: "wink",
  product: "sit",
  checkout: "heart",
  account: "heart",
  confirmation: "star",
};

/* Coco's commentary, one pool per page. The first line greets, the rest rotate. */
const COCO_LINES = {
  home: [
    "Welcome to Cordi Lab! I'm Coco ♡",
    "Psst… have you seen Soft Landing yet?",
    "Everything here stays facing you. Bunny promise!",
    "Coco's favorite spot? Right here, watching you browse.",
  ],
  "our-story": [
    "Ooh, story time! Coco loves this part.",
    "The zero-flip test? I supervised it myself.",
    "This page always makes me tear up a little ♡",
    "Every design gets a bunny stamp of approval.",
  ],
  collections: [
    "Pick me! Pick me! (I'm Coco, by the way!)",
    "One of these boxes hides a lucky variant…",
    "I helped wrap every single box ♡",
    "Blind boxes are Coco's favorite kind of surprise!",
  ],
  product: [
    "Coco knows which design is inside… but won't tell!",
    "Which one will land with you?",
    "The whole set means no bunny gets left behind ♡",
    "The wing motifs? Totally Coco's idea.",
  ],
  checkout: [
    "Almost there, friend! Coco is cheering for you!",
    "I'll guard your cart while you type ♡",
    "Coco's tip: double check that card number!",
    "Your blind box is getting so excited!",
  ],
  account: [
    "Welcome to your burrow!",
    "Coco Points add up fast, promise ♡",
    "Every 100 points is a dollar off. Coco math!",
  ],
  confirmation: [
    "Yippee! Coco will wave your box goodbye personally!",
    "Come back and visit Coco soon, okay?",
    "I wonder which design you'll meet… eee!",
  ],
};

let introduced = false;
let lastLine = "";
let chatterTimer = null;

function cocoIntroLine(key) {
  const pool = COCO_LINES[key] || COCO_LINES.home;
  if (!introduced) {
    introduced = true;
    return "Hi! I'm Coco, the Cordi Lab bunny ♡";
  }
  return pool[0];
}

function cocoChatterLine(key) {
  const pool = COCO_LINES[key] || COCO_LINES.home;
  const options = pool.filter((l) => l !== lastLine);
  return options[Math.floor(Math.random() * options.length)];
}

function scheduleChatter() {
  clearTimeout(chatterTimer);
  chatterTimer = setTimeout(() => {
    if (!asleep && !chatOpen && !document.hidden && Date.now() >= poseLock) {
      const line = cocoChatterLine(currentRouteKey);
      lastLine = line;
      sayBubble(line, 4200);
    }
    scheduleChatter();
  }, 13000 + Math.random() * 9000);
}

let currentRouteKey = "home";

function resolveRoute() {
  const hash = location.hash.replace(/^#\/?/, "").replace(/\/+$/, "");
  if (hash === "" || hash === "home") return "home";
  if (hash === "our-story") return "our-story";
  if (hash === "collections") return "collections";
  if (hash === "collections/soft-landing") return "product";
  if (hash === "checkout") return "checkout";
  if (hash === "account") return "account";
  if (hash === "confirmation") return "confirmation";
  return "home";
}

function render() {
  const key = resolveRoute();
  currentRouteKey = key;
  const route = ROUTES[key];

  $("#app").innerHTML = route.render();
  if (route.bind) route.bind();

  $$(".site-nav a").forEach((a) =>
    a.classList.toggle("active", a.dataset.nav === route.nav)
  );

  scrollTo({ top: 0, behavior: "instant" });
  closeCart();

  renderAccountUI();
  poseLock = 0;
  asleep = false;
  setMascotPose(ROUTE_POSE[key] || "sit", true);
  const line = cocoIntroLine(key);
  lastLine = line;
  sayBubble(line, 3200);
  wakeMascot();
  scheduleChatter();
}

addEventListener("hashchange", render);

/* ---------- Init ---------- */

renderCartUI();
render();
