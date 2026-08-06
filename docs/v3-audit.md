# Dataset v3 stratification audit

V3 assigns whole observation-context groups while constraining ambiguity, outcome-count,
action-family, scenario-family, and supported mechanic-tag distributions.

| Gate | Result |
| --- | ---: |
| Unique prompts | 1,525 / 1,525 |
| Prompt overlaps | 0 |
| Context overlaps | 0 |
| Ambiguity-rate max gap | 0.93% |
| Mechanic-share max gap | 5.54% |

## Split composition

| Split | Records | Context groups | Identifiable | Ambiguous | Ambiguity rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,218 | 154 | 731 | 487 | 39.98% |
| valid | 154 | 19 | 91 | 63 | 40.91% |
| test | 153 | 19 | 91 | 62 | 40.52% |

The current test split remains diagnostic because earlier experiments informed the V3
methodology. A newly generated untouched holdout is required for a final claim.
