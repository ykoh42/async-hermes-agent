# Async Hermes Agent Documentation

This site is built with [Docusaurus](https://docusaurus.io/) and published at
[ykoh42.github.io/async-hermes-agent](https://ykoh42.github.io/async-hermes-agent/).
It preserves the visual foundation of the upstream Hermes Agent documentation
while documenting this repository's native-async, library-focused surface.

## Local development

Node.js 26, npm 12, and Python 3.11 or newer are recommended.

```bash
cd website
npm ci
npm start
```

The prebuild step runs `scripts/generate-llms-txt.py` locally to keep
`static/llms.txt` and `static/llms-full.txt` synchronized with the checked-in
documentation. It does not fetch remote content.

## Verify a production build

```bash
cd website
npm run typecheck
npm run build
```

The static site is written to `website/build/`.

## Deployment

`.github/workflows/deploy-site.yml` builds and deploys `website/build/` to
GitHub Pages after documentation changes land on `main`. It can also be run
manually from `main` through GitHub Actions. No `gh-pages` branch or local
deployment command is required.

## Attribution

Async Hermes Agent is derived from
[Hermes Agent](https://github.com/NousResearch/hermes-agent) by
[Nous Research](https://nousresearch.com). The upstream theme and visual assets
are retained under the repository's MIT license.
