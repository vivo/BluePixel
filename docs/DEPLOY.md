# Deploying the BluePixel project page

Canonical public URL: <https://tech.vivo.com/BluePixel/>

## Static files

Use `docs/index.html` and the complete `docs/img/` directory as the source. Preserve their relative paths when serving them under `/BluePixel/`:

```
BluePixel/
├── index.html
└── img/
```

`docs/site/docs/` is a standalone copy for separate static hosting. If that copy is used, verify that its HTML and every referenced image match the canonical sources. Documentation and review materials do not need to be included in the deployed static directory.

## Hosting

Configure the approved hosting platform to serve these static files at the canonical URL. This repository does not contain a deployment job or configure the tech.vivo.com domain.

For an optional GitHub Pages preview, select the target repository branch and `/docs` folder in Settings → Pages. That preview is separate from the canonical public URL; do not assume it configures the company-hosted site. The `.nojekyll` files are included for this preview mode.

## Checks

- Verify every HTML and JavaScript image reference in the actual deployment directory.
- Test all before/after comparisons and example switches.
- Keep the paper link and BibTeX aligned with <https://arxiv.org/abs/2610.19882>.
- The waitlist button is a placeholder until a registration URL is provided.
- Google Fonts loads external font resources. GoatCounter is disabled in the source; enabling it changes the site's external requests and requires the actual site configuration.
