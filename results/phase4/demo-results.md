# Phase 4 measured demo results

Generated from completed browser exports by `python -m experiments.phase4_report`.
These are real CPU checkpoint responses on curated training examples.
Hypotheses and reflections in these acceptance records were written by the test author, not students.

## Evidence

Source: [evidence-acceptance-record.json](evidence-acceptance-record.json).

| Selection | Saliency sum / mean activity | Target margin Δ | Target probability Δ (points) |
| --- | ---: | ---: | ---: |
| patch (8,12) | 6.6464 | -0.0388 | +1.65 |
| patch (20,12) | 4.3445 | +1.2219 | +7.13 |

## Hidden

Source: [hidden-acceptance-record.json](hidden-acceptance-record.json).

| Selection | Saliency sum / mean activity | Target margin Δ | Target probability Δ (points) |
| --- | ---: | ---: | ---: |
| conv2 channel 0 | 1.1794 | +0.1341 | -0.26 |
| conv2 channel 1 | 0.1488 | -2.1566 | -8.57 |
| conv2 channel 5 | 0.0000 | +0.0000 | +0.00 |

## Shortcut

Source: [shortcut-acceptance-record.json](shortcut-acceptance-record.json).

| Background | Revealed model | Prediction | Target probability | Target margin |
| --- | --- | --- | ---: | ---: |
| yellow | shortcut | airplane | 95.93% | 3.3177 |
| yellow | balanced | airplane | 90.41% | 2.5044 |
| green | shortcut | cat | 47.22% | -0.0609 |
| green | balanced | airplane | 95.44% | 3.4942 |
| blue | shortcut | fish | 12.95% | -1.8837 |
| blue | balanced | airplane | 91.46% | 2.7678 |

| Revealed model | Target probability range (points) | Target margin range | Any label flip |
| --- | ---: | ---: | --- |
| shortcut | 82.99 | 5.2013 | yes |
| balanced | 5.03 | 0.9898 | no |

## What these examples support

Compare local sensitivity with the measured effect of a larger input change; compare average channel activity with its measured ablation effect; compare backgrounds with the foreground fixed.
No aggregate held-out result, unique semantic mechanism, or student learning gain follows from these three examples.
