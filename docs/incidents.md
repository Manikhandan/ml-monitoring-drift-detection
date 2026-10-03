# On-call model serving

Code owners: platform.

## High inference error rate

1. Hit `/ready`.
2. Inspect recent jsonl under `var/predictions.jsonl`.
3. Roll back by promoting the previous version.

## Drift ticket

1. GET `/v1/drift`.
2. If `alert` is true, stop Production promotions.
3. Run `python scripts/train.py` on a new window after reviewing the feature histograms.
