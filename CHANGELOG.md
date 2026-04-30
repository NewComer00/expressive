# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

<!-- version list -->

## v0.10.0 (2026-04-30)

### Features

- **expressions**: Add BrecLoader and VoicLoader for Breathiness and Voicing
  ([`55fe4f4`](https://github.com/NewComer00/expressive/commit/55fe4f4eb9d92946fdff53f6117091b9271b49e6))

- **expressions, embedder, wavtool**: Add Breathiness & Voicing params, mHuBERT backend
  ([`55fe4f4`](https://github.com/NewComer00/expressive/commit/55fe4f4eb9d92946fdff53f6117091b9271b49e6))


## v0.9.1 (2026-04-24)

### Bug Fixes

- **ustx**: Catch SameFileError when same-path input/output; remove hardcoded SUPPORTED_EXPRESSIONS
  restriction; add allow_unicode=True to oyaml.dump; warn when no expression points fit any voice
  part
  ([`96a8d29`](https://github.com/NewComer00/expressive/commit/96a8d29e4d92e7f19496d27ea75196f113ed997a))

- **ustx, seqtool, gui**: Fix in-place editing, interpolation, and unicode
  ([`96a8d29`](https://github.com/NewComer00/expressive/commit/96a8d29e4d92e7f19496d27ea75196f113ed997a))

- **wavesurfer**: Add loading indicator and error handling
  ([#28](https://github.com/NewComer00/expressive/pull/28),
  [`e257918`](https://github.com/NewComer00/expressive/commit/e257918a23df77326a4369d4d54aa31cab01174f))

- **wavesurfer**: Hide loader and error msg by on load
  ([`a153c8e`](https://github.com/NewComer00/expressive/commit/a153c8ecedd0d6c59a83423c6c23b10f99563d13))


## v0.9.0 (2026-04-15)

### Bug Fixes

- **pitd**: Fix overly flat PITD curves (issue #21); special thanks to @ma0shu for helping identify
  and diagnose this critical bug ❤
  ([`6f4bd54`](https://github.com/NewComer00/expressive/commit/6f4bd545613bed2217f0cf22f1e2614e48a4f7dd))

### Documentation

- **readme**: Add v0.9.0+ warning; document scaler change; update troubleshooting
  ([`6f4bd54`](https://github.com/NewComer00/expressive/commit/6f4bd545613bed2217f0cf22f1e2614e48a4f7dd))

### Features

- **f0**: Add hybrid F0 backend with fallback; improve stability and reduce discontinuities
  ([`6f4bd54`](https://github.com/NewComer00/expressive/commit/6f4bd545613bed2217f0cf22f1e2614e48a4f7dd))

- **pitd, f0, docs**: Improve PITD and add hybrid F0 backend
  ([`6f4bd54`](https://github.com/NewComer00/expressive/commit/6f4bd545613bed2217f0cf22f1e2614e48a4f7dd))

### Breaking Changes

- **pitd, f0, docs**: PITD scaler default changed from 2.0 to 1.0; PITD results prior to v0.9.0 are
  unreliable


## v0.8.0 (2026-04-11)

### Features

- Add rmvpe-onnx as default pitch extraction backend
  ([`86a152b`](https://github.com/NewComer00/expressive/commit/86a152b695d19b08ac1ae0a961ad44f33aa55f3a))


## v0.7.0 (2026-04-01)

### Bug Fixes

- **test_i18n**: Add create=True to ctypes.windll patches for Linux CI compatibility
  ([`2a0b90d`](https://github.com/NewComer00/expressive/commit/2a0b90d5a240f5c2e567243ac0a232e3a297a10e))

### Documentation

- **README,README.en**: Add Data Viewer screenshot alongside workflow
  ([`23bb533`](https://github.com/NewComer00/expressive/commit/23bb533308929100a8660d7ca55b60cbf5c2e4f1))

### Features

- **build**: Replace inline PyInstaller flags with
  ([`23bb533`](https://github.com/NewComer00/expressive/commit/23bb533308929100a8660d7ca55b60cbf5c2e4f1))

- **ui**: Add ClosableTabs component extending ui.tabs with per-tab
  ([`23bb533`](https://github.com/NewComer00/expressive/commit/23bb533308929100a8660d7ca55b60cbf5c2e4f1))

- **viewer**: Add `expressive-viewer` — real-time expression curve
  ([`23bb533`](https://github.com/NewComer00/expressive/commit/23bb533308929100a8660d7ca55b60cbf5c2e4f1))

- **viewer,build,ui,docs**: Add expression curve viewer, spec-based build, and closable tabs
  ([`23bb533`](https://github.com/NewComer00/expressive/commit/23bb533308929100a8660d7ca55b60cbf5c2e4f1))

### Refactoring

- **wavtool**: Defer heavy imports (sklearn, skimage, soundfile,
  ([`23bb533`](https://github.com/NewComer00/expressive/commit/23bb533308929100a8660d7ca55b60cbf5c2e4f1))

- **worker**: Change `lang` field type from `str` to
  ([`23bb533`](https://github.com/NewComer00/expressive/commit/23bb533308929100a8660d7ca55b60cbf5c2e4f1))


## v0.6.0 (2026-03-22)

### Bug Fixes

- **wavtool**: Extract_wav_frequency now returns np.ndarray instead of
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

### Documentation

- **readme**: Improve demo presentation and add collapsible video
  ([`b6a64e0`](https://github.com/NewComer00/expressive/commit/b6a64e019222559cbcf1a8b4b0acd6a43d3303d3))

### Features

- **base**: Register tick converters from ustx_time_axis in
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **dyn,pitd,tenc**: Drop tempo/ppqn kwargs from align_sequence_tick
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **examples**: Add テトリス and Прекрасное Далеко examples with
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **seqtool**: Replace time_to_ticks/ticks_to_time with a
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **ustx**: Add TimeAxis — piecewise tick ↔ ms converter replicating
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **ustx**: Add UProject, UVoicePart, UCurve, UTrack, UTempo,
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **ustx**: Add UstxEditor — RAII context manager with exclusive
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **ustx,seqtool,base**: Replace flat-dict USTX API with typed datamodel and multi-tempo support
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

### Testing

- **base**: Replace loader.tempo with ustx_time_axis assertions; remove
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **expressive**: Fix integration test to use UProject attribute access
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **seqtool**: Replace TestTimeConversion with TestTickConverterRegistry
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **ustx**: Replace dict-based assertions with UProject attribute
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

- **wavtool**: Fix list → ndarray type assertions for
  ([`83d33f0`](https://github.com/NewComer00/expressive/commit/83d33f0c766c9e55ebd66e7e6065c1f1ad723ba3))

### Breaking Changes

- **ustx,seqtool,base**: Load_ustx() now returns UProject instead of dict; save_ustx() now accepts
  UProject instead of dict; edit_ustx_expression_curve() is removed — use UstxEditor or
  UVoicePart.set_curve() instead.


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
