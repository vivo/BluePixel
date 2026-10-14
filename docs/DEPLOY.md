# Deploying the BluePixel project page to GitHub Pages

## What to upload

Push these files to the `vivo/BluePixel` repo (keep the relative structure):

```
index.html
teaser.jpg
img/            (all jpgs — gallery figures + real arcade input/output pairs + prompt-arcade outputs)
.nojekyll       (already created; disables Jekyll processing)
```

`preview.png` and this file do not need to be deployed.

## GitHub Pages setup

1. Repo **Settings → Pages**.
2. Source: *Deploy from a branch* → pick `main` and either `/ (root)` or `/docs`.
   (If you prefer `/docs`, move the four items above into a `docs/` folder.)
3. The page goes live at `https://vivo.github.io/BluePixel/` within a few minutes.
4. Optional: add a CNAME file for a custom domain (e.g. `bluepixel.vivo.com.cn/page`).

## Visitor tracking — two layers

### Layer 1 (zero setup): GitHub Insights → Traffic
On the repo page, **Insights → Traffic** shows page views and unique
visitors for the last 14 days (both the repo and the GitHub Pages site).
Anonymous aggregate numbers only — no identities. This is the native
GitHub answer for "how many people visited".

### Layer 2 (one line): GoatCounter — referrers, countries, per-page views
1. Register the site at <https://www.goatcounter.com> (free for
   non-commercial; use the final URL `vivo.github.io/BluePixel`).
2. Open `index.html`, find the block marked
   `Visitor analytics (GoatCounter ...)`, uncomment the `<script>` line and
   replace `YOURCODE` with your site code.
3. Stats appear at `https://<YOURCODE>.goatcounter.com` — cookieless and
   GDPR-friendly.

Note: no analytics service can tell you *who* visited (name/identity);
everything above is anonymous counts. Identifying individuals would require
server logs, which GitHub Pages does not expose.

## After the arXiv number exists

- `index.html`: point the PAPER button at the arXiv abs page.
- `index.html`: fill the arXiv id into the BibTeX block.
- Optional: add `og:url` with the final page URL.
