# DREAM preprint: counting conventions and archived-result provenance

This documentation clarification applies to benchmark snapshot `84a2798cd46565b3dd72f9210511a7f5f7f1cf1e`. Add it to the documentation on the current branch; do not rewrite historical benchmark files, move the existing tag, or modify archived CSV values to make older examples agree.

## Headline versus raw counts

| World | Raw baseline / DREAM | Raw count reduction | Complete baseline / DREAM | Complete-count reduction |
|---|---|---|---|---|
| A | 9 / 2 | 77.8% | 11 / 3 | 72.7% |
| B | 26 / 8 | 69.2% | 28 / 9 | 67.9% |
| C | 33 / 3 | 90.9% | 35 / 4 | 88.6% |

The preprint headline uses the complete, boundary-inclusive waypoint sets under the same seven-field accounting policy. The raw counts above concern the stated regularized two-rate reproduction on the selected clean recording; repeated recordings and noise realizations are not independent path geometries. Shared field counts do not imply identical payload semantics or measured ROS 2 network traffic.

Older README values such as 89.5/69.2/90.9% are legacy diagnostics, not the current complete-count headline. The World A discrepancy also involves the unregularized/regularized yaw-test convention, not only boundary insertion. Keep legacy results only if labelled with their actual baseline and counting policy. Replace the unqualified headline claims in README section 2.3.1 and its final checklist; merely appending this note while leaving contradictory labels is insufficient.

`--compare` reports a raw-output count ratio, `1 - len(d_idx) / len(b_idx)`. Do not present its console output as the complete-count headline unless the wrapper explicitly constructs the complete sets and uses the stated baseline settings. Under nominal noise, the reported World A mean complete counts are 20.0/3.0, giving approximately 85% from the ratio of displayed means, not necessarily the mean of per-run reduction ratios.

## Completion is not true-position accuracy

An estimate-frame endpoint of 0.030 m denotes controller stopping-tolerance completion. It is not a measurement of physical return accuracy. The two-state protocol reports PE_true separately and uses the stated 0.06 m operational criterion. Corridor driver endpoints at a 0.06 m odometry stop, offline kinematic clearances and physics bumper-contact records remain separate evidence groups.

## Noise preprocessing

In the forward reconstruction used by the reference noise protocol, `reconstruct_noisy` clips perturbed linear velocity at zero before integration: `vn = max(0.0, v + sigma_v * eta)`. Gaussian refers to the perturbation before clipping. This is benchmark preprocessing, not a restriction of the general sampler's signed-progress definition. Do not infer return-phase clipping from this forward function; inspect the identified two-state return engine separately.

## Archived per-run CSV versus printed summary

Archived records: `bench_data/two_state_runs.csv` at commit `84a2798cd46565b3dd72f9210511a7f5f7f1cf1e`.

https://github.com/dungltr/DreamRobot/blob/84a2798cd46565b3dd72f9210511a7f5f7f1cf1e/bench_data/two_state_runs.csv

The noise-robustness CLI reports aggregate evaluation results:

```sh
python3 run_gazebo_benchmark.py --noise-robustness --runs 50
```

The option names are recorded for documentation, not executed by this revision. Do not describe this command as a direct CSV exporter or byte-identical regeneration procedure without the exact export script, configuration and evidence. If an exporter was used but is not archived, document that provenance gap rather than inventing a filename.

The documented CSV fields include `world`, `sampler`, `seed`, `return_seed`, `N_sampler`, `N_stored`, `PE_true`, `PE_odom`, `completed`, `timed_out`, and `forward_drift`. The reference protocol specifies 50 realizations per world/sampler; validate the archived file locally before asserting its row count or reconstructing statistics. The supplied preflight utility can inspect an existing clone and CSV without importing or running the benchmark.

Use the immutable commit as the artifact identifier. Check the peeled commit of `dream-ieee-revision-v35`, not only an annotated-tag message or historical release description. A documentation update should be a new commit on the current branch, not a moved benchmark tag. Public repository access does not itself establish a code/data license; the owner must select appropriate permissions, including for third-party material.
