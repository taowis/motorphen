# E1 implementation spec

What the GenGait paper (Motta et al., arXiv:2604.01997v1, sections 2.2 to 3.2)
specifies, and what we had to choose because the paper leaves it open. Section
and equation numbers refer to arXiv v1; check against the journal version.

## Specified by the paper (followed as written)

| Item | Paper |
|---|---|
| Joints | 12: neck, shoulders, elbows, pelvis, hips, knees, ankles |
| Window | 7 frames, stride 1, 84 joint-frame tokens |
| Token features | 12-d: sin and cos of three Euler angles, plus 6D rotation (first two columns of the rotation matrix from the same angles) |
| Input embedding | linear projection; fixed sinusoidal positional code P; learned joint, frame and motion (angular velocity) embeddings E |
| Encoder / decoder | 8 / 2 Transformer layers, 12 heads, head dim 24, d = 288, dropout 0.1 |
| Masking | masked tokens replaced by a learned [MASK]; decoder input = [MASK] sequence with visible encoder memory at visible positions, P and E added again |
| Output used at inference | reconstructed last-frame tokens |
| Optimiser | AdamW, lr 2e-4, betas (0.9, 0.95), weight decay 0.05 except biases and norms, grad clip 1.0, 250 epochs, batch 256 |
| Training masks | curriculum from random joint dropout (50%) to structured, biomechanically plausible patterns (e.g. whole limb), transition complete by epoch 60; temporal spans: 25% single frame, 25% full window, 50% 2 to 6 frames, start uniform, same keep-set within the span, frames outside sampled independently |
| Losses (equal weights, on sin-cos) | (i) L1 at the final frame, (ii) masked tokens over the window, (iii) visible tokens over the window, (iv) angular velocity consistency, (v) context invariance across different masks |
| Pass 1 at inference | 6 tiles, each hiding one of neck, pelvis, L/R hip, L/R knee, plus an unmasked baseline tile |
| Badness score | Eqs. 1 to 6: C_ROM = sum_i w_i clip(abs(wrap(dphi_i)) / ROM_i, 0, 1); E_geom = (1 - cos theta) / 2 between baseline and tile bone vectors; B_j(t) = E_geom (0.5 + 0.5 C_ROM); trial score = PeakStat_t B_j(t) |
| Selection | top K = 2 joints whose trial score exceeds a calibrated noise floor |
| Pass 2 | re-run with only the flagged joints hidden from the encoder |
| Evaluation (sec. 3.2) | right-side pelvis flex/ext, hip ab/ad, hip flex/ext, knee flex/ext; cycles time-normalised to 100 frames; mean per participant-condition; RMSE to the training mean mu(t); normative band mu +/- 2 sigma |

## Not specified: our choices

| Item | Our choice | Reason |
|---|---|---|
| Euler convention | Plug-in Gait Cardan angles as delivered (flexion, ab/adduction, rotation; Y-X-Z) instead of intrinsic XYZ from markerless keypoints | the angles come from the motion capture model; 6D built from the same angles as in the paper |
| Positional code P | sinusoidal over the flattened token index (joint x 7 + frame) | simplest reading |
| Motion embedding | linear map of the per-token wrapped angular velocity (3 values, zero at frame 0) | "learned embedding of instantaneous angular velocity" |
| Curriculum schedule | epochs 0 to 4 random only; 5 to 59 probability of structured rises linearly; from 60 on 90% structured, 10% random (read from Fig. 3) | Fig. 3 marks epochs 0, 5, 60, 250 and 10% / 90% |
| Structured patterns | one of: left arm, right arm, left leg (hip, knee, ankle), right leg, trunk (neck, pelvis), one random joint, two random joints | "e.g. full limb removal" |
| Velocity loss | L1 between predicted and true frame-to-frame differences of sin-cos | |
| Context-invariance loss | second mask drawn for the same window; L1 between the two final-frame predictions | |
| Bone vector for E_geom | unit vector of the child segment in the parent frame, R(angles_j) applied to the segment's long axis; no full forward kinematics | joint positions are not part of our input |
| ROM_i | 1st to 99th percentile range of each joint axis in the training subjects | paper says "anatomical ROM limits" without values |
| w_i | sagittal 0.6, frontal 0.25, transverse 0.15 for all joints | paper gives no values |
| PeakStat | 95th percentile over frames | "peak statistic" not defined; plain max is sensitive to single-frame spikes |
| Noise floor | per joint, 99th percentile of trial scores in held-out healthy validation subjects | "calibrated noise floor" not defined |
| Gait cycles | from Plug-in Gait foot strike events, not heel-height peaks | events are in the C3D files |
| Pass 2 output | reconstructed last frame of each window for all joints, not only the flagged ones; first 6 frames keep the input | the paper says joints outside the top K are "implicitly constrained by the normative prior", which implies the full reconstruction is used |
| Normative mean mu(t) | all training trials (the dataset has self-selected speed only) | paper used normal-speed trials |
| Selection rule reported | both the paper's rule (above noise floor, at most K) and always top K | the second is a stricter specificity check |
| Split | by subject: 110 train, 8 validation (noise floor, monitoring), 17 test, from the 135 full-body subjects; seed 20261103 | paper used 150 / 10 |

Any change to a choice above is recorded here with the date.
