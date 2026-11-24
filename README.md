# GraphNNSol — valve controls and rewrite geometry

Follow-up to [GraphNN](https://github.com/anttiluode/GraphNN) and the November 24, 2026 Neon audit. The audit found signed scalar transport tied complex phase on all mark tests, with lower chained B MSE (0.018213 vs 0.024474). This does not isolate sign from attenuation or normalization.

## Gate 1 (preregistered, no results yet)

Keep the upstream 8x8 graph, eight channels, four steps, image writer, ping, decoder, loss and procedural data. Train with the original A/B/A' protocol, then test role swap, independent replacement and alternating rewrites on disjoint held-out seeds.

| Arm | Transport | Persistent bytes |
|---|---|---:|
| positive | [0,1] attenuation, fixed degree | 896 |
| signed | [-1,1] gain, fixed degree | 896 |
| binary | exactly -1 or +1, fixed degree | 112 |
| phase | unit-norm conjugate rotations | 896 |
| node | signed source-node modulation, orientation-specific, both components | 896 |
| slots | fixed disjoint address slots, direct read | 896 |
| binary_slots | same slots with 1-bit values | 112 |

The graph arms have equal shared parameter counts and use the same writer and readout. Slot readout uses the same learned parameter count but not equal FLOPs: it is an ordinary addressed-memory reference, not a transport-only ablation. Binary arms keep the same number of edge variables but 1/8 the persistent storage; comparison at equal bits with a smaller 8-bit graph is a separate follow-up. No hidden per-example float state survives a quantized write. The straight-through estimator is used in training; evaluation explicitly round-trips through uint8 or packed bits at every write.

Primary outcome: chained reconstruction MSE of both images, and joint mark recovery, per seed. Secondary: B drift after A rewrite, reciprocal drift, real vs sham update size. Geometry: compare JVP of both outputs along actual and sham write directions with magnitude-matched signed-permutation nulls. Report finite changes too (binary null matches flip counts conditional on original bit; 8-bit null may clip, report achieved norm). The binary JVP is only a smooth relaxation. Small directional derivative is evidence of local protection, not proof of a global null space.

Predeclare interpretations:
- Positive ties signed: signs have not earned their complexity.
- Binary ties signed within 0.002 absolute B chain MSE, 1 percentage point joint recovery, and 0.002 A chain MSE on the mean: sign-only memory is competitive at 1/8 storage. Also inspect every seed; no significance claim from three seeds.
- Signed beats positive, binary trails: signed attenuation may matter; binary optimization remains a confound.
- Signed ties node: edge-specific transport has not earned a benefit over this node baseline.
- Slots win: do not claim a distributed-memory advantage.
- Rewrite B JVP energy per delta is not smaller than matched nulls: no evidence the writer preferentially follows protected directions. A and B finite error also matter; compare sham separately.

## Run on Windows

With the existing PyTorch CUDA environment:

```bat
git clone https://github.com/anttiluode/GraphNNSol
cd GraphNNSol
python -m pip install numpy pillow
python gate1.py --self-test
python gate1.py --device cuda
```

Quick first pass: `python gate1.py --device cuda --steps 20 --seeds 41 --geometry-count 2 --output runs/smoke`

Full run writes `runs/gate1/summary.json` after each arm, and model checkpoints. Use a different output directory for changed protocols. CPU works too. No external dataset or cloud needed. `--modes positive signed binary phase` runs only the primary valve comparison.

Upstream files copied from GraphNN (MIT, original LICENSE retained). This is an empirical investigation, not an application or verification of the VP/VNP theorem. The analogy is measuring what a representation preserves under changes and resource constraints.
