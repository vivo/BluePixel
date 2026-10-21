# BluePixel — Project Page

Official project page: <https://tech.vivo.com/BluePixel/>

Paper: <https://arxiv.org/abs/2610.19882>

## Source layout

- `docs/index.html` and `docs/img/` are the canonical site sources.
- `docs/site/docs/` is a standalone copy for separate static hosting. Keep its HTML and referenced images synchronized with the canonical sources.
- `docs/.nojekyll` and `docs/site/docs/.nojekyll` disable Jekyll processing when using GitHub Pages.
- Hosting the sources does not configure or change the canonical domain. See [deployment notes](../DEPLOY.md).

## Before deployment

- Check that every image referenced by HTML and inline JavaScript exists under the selected source directory.
- Keep the PAPER link and BibTeX aligned with arXiv:2610.19882.
- Replace the waitlist placeholder only when a registration URL is available.
- GoatCounter remains disabled; its snippet is commented out in both HTML copies.
