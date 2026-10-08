# configs

Tracked configuration: dataset and model cards arrive here in build phase 2
(`configs/datasets/*.yaml`, `configs/models/*.yaml`).

`settings.yaml` (your local settings) is git-ignored and must never hold
credentials; see `core/common/config.py` for the keys it accepts.
