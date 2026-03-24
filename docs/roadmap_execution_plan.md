# Roadmap Execution Plan

This document translates the current GitHub issue backlog into a practical execution order.

It is intended to answer two questions:

1. What should be fixed first?
2. Which work depends on earlier stabilization tasks?

## Phase 1: Core Stabilization

Goal: make the package installable, testable, and usable through the main runtime paths.

1. [#7](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/7) Fix package layout so the project installs and imports as `wifi_activity_recognition`
2. [#8](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/8) Stop relying on test import shims and make CI validate a real package install
3. [#12](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/12) Unify model serialization across training, inference, benchmark, and export flows
4. [#13](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/13) Implement or remove missing helper APIs referenced by the CLI
5. [#14](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/14) Fix the predict CLI path to use the actual CSI loader API
6. [#17](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/17) Redesign the evaluate CLI contract to match the dataset loader and result export APIs
7. [#18](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/18) Refactor the collect command to use one continuous stream session

## Phase 2: API and Documentation Alignment

Goal: align the public surface with reality, deepen validation, and clean up what only becomes visible after install/import is fixed.

1. [#19](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/19) Derive CLI hardware choices from the registered hardware drivers
2. [#11](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/11) Standardize Python version support and expand the CI test matrix
3. [#6](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/6) Reconcile CLI, README, and docs with the commands that actually exist
4. [#15](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/15) Make README and package-level example code match the real public API
5. [#16](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/16) Rewrite the training guide so it matches the implemented dataset and trainer APIs
6. [#20](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/20) Strengthen preprocessing and feature-extraction validation beyond smoke tests
7. [#21](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/21) Add deeper correctness validation for model behavior across supported architectures
8. [#23](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/23) Audit and improve modules whose current coverage is mostly compile-level or smoke-level
9. [#24](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/24) Define a performance regression strategy for latency, memory, and accuracy-sensitive paths
10. [#26](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/26) Plan post-install architectural cleanup after the package layout fix lands

## Phase 3: Product and Onboarding Improvements

Goal: improve credibility, onboarding, real-device confidence, and deployment readiness.

1. [#10](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/10) Align Broadcom/MediaTek hardware support claims with actual implementation
2. [#9](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/9) Implement a real quickstart/demo workflow for first-time users
3. [#22](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/22) Establish real-device validation coverage for hardware drivers
4. [#25](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/25) Verify deployment paths end to end across Docker, edge runtime, and Kubernetes assets

## Recommended Global Execution Order

This is the recommended cross-phase order when executing work sequentially:

1. [#7](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/7)
2. [#8](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/8)
3. [#12](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/12)
4. [#13](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/13)
5. [#14](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/14)
6. [#17](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/17)
7. [#18](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/18)
8. [#19](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/19)
9. [#11](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/11)
10. [#6](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/6)
11. [#15](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/15)
12. [#16](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/16)
13. [#20](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/20)
14. [#21](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/21)
15. [#23](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/23)
16. [#24](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/24)
17. [#26](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/26)
18. [#10](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/10)
19. [#9](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/9)
20. [#22](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/22)
21. [#25](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/25)

## Notes

- Phase 1 should be completed before treating documentation, deployment, or benchmark claims as trustworthy.
- Phase 2 is where correctness depth and architectural cleanup become worthwhile, because the package will behave more like a normal installed project.
- Phase 3 should focus on real-world credibility: onboarding, hardware validation, and deployment verification.
