# Changelog

## v0.2 - 14 November 2026

This update extends the initial v0.1 source release.

### Added

- DFL initialization and forward projection in the UDEH module.
- Learnable Scale initialization and forward processing.
- Conv_GN initialization.
- Central-difference convolution initialization and kernel construction.
- Horizontal-difference kernel construction.

### Updated

- Package version to 0.2.0.
- Release manifest and scope documentation to match the available implementations.
- README links to version history and the version comparison.

### Verification

`python -B verify_release.py` passed: 8 regression tests passed and 2 were skipped because the corresponding functionality is scheduled for a later release.

See [RELEASE_SCOPE.md](RELEASE_SCOPE.md) for available and scheduled functionality.

[Compare v0.1 with v0.2](https://github.com/flamefover/MED-Net/compare/v0.1...v0.2)

## v0.1 - Initial source snapshot uploaded 23 October 2026

- Initial architecture configuration, available module implementations, documentation, and regression checks.
- Original commit: [1cff35e](https://github.com/flamefover/MED-Net/commit/1cff35e43f2fd9af83912d185520877d76e8167e).
- The v0.1 tag and release entry were added on 14 November 2026 to make the existing history easier to navigate; the original commit and code were preserved unchanged.
