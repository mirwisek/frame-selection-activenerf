# Logbook of Experiments

All notable changes within this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [NeRF](https://www.matthewtancik.com/nerf). 

## [1.0.1] - 2024-01-16

### Added

- Added `subject_results` that contains results for test of our approach on other NeRF subjects than Lego Truck. The NeRF subjects are `800x800x4` instead of `100x100x3` that were scaled down.

### Fixed

- The deprecated warnings: Clonning of a tensor, the default parameter for ResNet18.
- Documented Floating Point Precision Problem [#1](https://github.com/mirwisek/frame-selection-activenerf/issues/1).

## [1.0.0] - 2024-01-16

### Added

- Initial release