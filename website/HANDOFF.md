# Cordi Lab — project handoff

Everything for the Cordi Lab store lives in this `website/` folder on the
branch `claude/product-website-4i70e8`. The repo's root (`alpha_evolver.py`
etc. on `main`) is an unrelated trading project — ignore it for store work.

## Live URLs

- Store: https://0103-parv.github.io/alpha-evolver/
- 3D design review space: https://0103-parv.github.io/alpha-evolver/design/
- Box packaging mockup: https://0103-parv.github.io/alpha-evolver/design/box/

## Layout

- `index.html`, `styles.css`, `app.js` — the whole store. No build step.
  Hash-routed SPA: Home, Our Story, Collections, product page, cart,
  checkout, account. Run locally: `python3 -m http.server 8000` in this
  folder, open http://localhost:8000
- `img/` — gallery photos (gallery-1..5 = box + the four design posters),
  product renders (product-1..4), favicons
- `design/` — Three.js design review viewer (vendor libs included)
- `design/box/` — packaging design sheet (HTML, render with any browser)
- `marketing/` — promo videos + the generators that made them
  - `cordi-lab-promo.mp4` — 10s square 1080x1080
  - `cordi-lab-reel.mp4` — 10s vertical 1080x1920 (parachute intro)
  - `scene.html` / `scene-reel.html` — deterministic animation pages;
    call `seek(t)` per frame, screenshot at 30fps, stitch with ffmpeg
  - `coco-*.png` — mascot sprites cut from the official pose sheet
- `HANDOFF.md` — this file

## Product facts (current)

- Series: Soft Landing. Designs: Silver Star, Snow Angel, Midnight Flight,
  plus hidden/lucky variant Teddy Wing (beige). 3 regular + 1 lucky = 4.
- Prices: single blind box $7.99, whole set (3 boxes, one of each regular
  design) $19.99. Defined once in `PRODUCTS` at the top of `app.js`.
- Free shipping threshold $40, flat $4.99 (also in `app.js`).
- Mascot: Coco. Clickable chat widget with keyword FAQ answers
  (`CHAT_TOPICS` in `app.js`).
- Accounts + "Coco Points" loyalty: browser-local (localStorage), password
  hashes via SHA-256. Earn 10 pts/$1, redeem 100 pts = $1 at checkout.
  Real cross-device accounts need a backend (Firebase/Supabase) later.

## Payments — CURRENT STATE AND NEXT STEPS

- Checkout has TWO paths: a Stripe payment-link button (real) and a demo
  card form below it (simulated, charges nothing).
- BOTH products have TEST-MODE Stripe links wired in `PRODUCTS` in
  `app.js` (`buy.stripe.com/test_...`). Test card 4242 4242 4242 4242.
  Both allow the customer to adjust quantity on the Stripe page.
- These live in the Stripe "CordiLabs sandbox"
  (acct_1TqbYLEAqLOYzAks, login: Parv's Stripe account), products
  "Soft Landing Blind Box" $7.99 and "Soft Landing Complete Set" $19.99.
  NOTE: an older link (in `git log` for app.js) pointed at a different,
  orphaned sandbox and wrongly bundled BOTH products ($27.98) — don't
  reuse it.
- TODO for real money (owner only — needs identity + bank):
  1. In the Stripe dashboard, "Switch to live account" → verify the
     business (identity + bank). Until then the account can't even exit
     sandbox mode.
  2. Recreate the two products + payment links in LIVE mode (no `test_`
     in URL) and swap the two `stripe:` URLs in `PRODUCTS`.
- Stripe links are public URLs — safe to commit.

## Deploying

The live site is the `gh-pages` branch: this folder's CONTENTS at the
branch ROOT (index.html at top level, plus img/, design/). To deploy:
copy the contents of `website/` over the gh-pages root, commit, push.
GitHub Pages rebuilds in ~30s ("pages build and deployment" in Actions).
Cache busting: bump `?v=N` on styles.css/app.js in index.html whenever
they change.

## Other open items

- Domain: cordilab.com must be purchased (any registrar). Then add DNS
  records + a CNAME file on gh-pages and set the custom domain in repo
  Settings → Pages.
- Real product photos will eventually replace the concept renders in
  `img/` (same filenames = zero code changes).
- Gallery caption says "Concept renders. Production photos coming soon!" —
  edit in `pageProduct()` in `app.js` if messaging changes.
