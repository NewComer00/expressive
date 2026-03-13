# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

<!-- version list -->

## v0.5.0 (2026-03-13)

### Bug Fixes

- **log**: Mark __main__ block as pragma: no cover
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

- **ui**: Fix WaveSurfer load/ready sequencing and zoom sync
  ([`1c2b201`](https://github.com/NewComer00/expressive/commit/1c2b2010ec6c0d75eaddc1ec8a8ad83cdfb532d5))

- **wavesurfer**: Prevent spurious full-width region on scroll-to-start
  ([`d540596`](https://github.com/NewComer00/expressive/commit/d54059600c6628e4f8b067e13615e1ebb8c7d52e))

### Code Style

- **ui**: Increase waveform scrollbar height from 6px to 10px
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

### Features

- Add audio trimming, waveform range selector, and tick offset fix
  ([`125a99f`](https://github.com/NewComer00/expressive/commit/125a99ffe66e33c86ee8a9b6b63ee7d9ce0d2816))

- **cli**: Extract add_expression_args_group and str2bool into utils/cli
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

- **dyn,tenc**: Add trim_silence option to mask leading/trailing silence
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

- **gui**: Add Trim Silence toggle to dyn and tenc expression cards
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

- **wavtool**: Move extract_wav_mfcc, extract_wav_frequency, extract_wav_rms to utils/wavtool
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

### Testing

- Add test_worker.py covering WorkerContext dataclass and setup_worker_context
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

- **cli**: Add tests for str2bool and add_expression_args_group; expand formatter coverage
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))

- **wavtool**: Add tests for extract_wav_mfcc, extract_wav_rms, extract_wav_frequency,
  get_wav_end_ts, ClampedWav
  ([`b58722a`](https://github.com/NewComer00/expressive/commit/b58722a045c2d8389b432d8989ebf423424144c1))


## v0.4.0 (2026-03-05)

### Features

- **gui**: Offload expression processing to subprocess with logging bridge
  ([`e01adf6`](https://github.com/NewComer00/expressive/commit/e01adf6d57ebcc3e92d1b7bc01f821799f39fb2d))


## v0.3.0 (2026-03-03)

### Bug Fixes

- **expressions/pitd**: Adjust swift-f0 defaults for confidence_ref and semitone_shift
  ([`e4a707c`](https://github.com/NewComer00/expressive/commit/e4a707c800472d4ebf51e44d70af8bb33138cae9))

### Features

- Add swift-f0 backend, polish help texts, update i18n
  ([`7f30d6c`](https://github.com/NewComer00/expressive/commit/7f30d6c5ddc7a2475aceacbae501b16f3ff973e6))

### Refactoring

- **cli**: Extract RichHelpFormatter to dedicated module
  ([`0fce1fd`](https://github.com/NewComer00/expressive/commit/0fce1fd38d7281d32e99b34cbad9b052c20549d3))


## v0.2.4 (2026-02-25)

### Bug Fixes

- **packaging**: Guard find_spec against missing parent package
  ([`7daccd2`](https://github.com/NewComer00/expressive/commit/7daccd2d5f50a00ea6efadde411d13b2c1997c34))


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
