# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

<!-- version list -->

## v0.2.3 (2026-02-25)

### Bug Fixes

- **packaging**: Add CPU and GPU installer variants
  ([`5fb0f1f`](https://github.com/NewComer00/expressive/commit/5fb0f1fc463f0a2291de535e1966ef45ca3d3311))


## v0.2.2 (2026-02-25)

### Bug Fixes

- **config**: Fix version_variables typo and sync __version__.py
  ([`058871d`](https://github.com/NewComer00/expressive/commit/058871d024c5603a61802fe8bf035e8f1a2df4ab))


## v0.2.1 (2026-02-25)

### Bug Fixes

- **build**: Compile .mo via hatch hook, untrack binary artifacts
  ([`d2141a0`](https://github.com/NewComer00/expressive/commit/d2141a0e00a05c676629ff811273c9302eb29eb1))

### Continuous Integration

- Pin all GitHub Actions to commit SHAs
  ([`d59a22e`](https://github.com/NewComer00/expressive/commit/d59a22e20e7a88338e6fceadbd9bbb3da43e20c7))


## v0.2.0 (2026-02-25)

### Features

- **gpu**: Make CUDA packages optional with skip_missing flag
  ([`d74ca13`](https://github.com/NewComer00/expressive/commit/d74ca13965cae508ba1459fb4a8c64e5b13a68b8))

- **gui**: Support root mode for wheel-installed app
  ([`399353b`](https://github.com/NewComer00/expressive/commit/399353b18b42f837e938197fdfcbf73e15935408))

### Refactoring

- **i18n**: Replace custom LazyString with lazy-string package
  ([`5af3b9e`](https://github.com/NewComer00/expressive/commit/5af3b9e4445893e8b4b11275d3f7b904f7a9b08b))


## v0.1.0 (2026-02-23)

- Initial Release
