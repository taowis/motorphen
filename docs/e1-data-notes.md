# E1 data notes

Findings while extracting the Van Criekinge 2023 C3D files (scripts/01_prepare_vc2023.py).

- Every walking trial has full-body Plug-in Gait angles. Some files prefix every
  point label with the subject name ("NAME:LHipAngles"); the reader strips it.
- Static calibration trials have no gait events and are skipped.
- 3 of 138 able-bodied subjects have missing or gappy upper-body angles, so
  that no segment keeps all 12 joints; they are usable for lower-limb models only.
- Whole walking trials are kept (not cut to first and last foot strike) because
  the healthy trials are short, about 3 s at 100 Hz.
- After resampling to 30 Hz: able-bodied 450 walking trials, 40,739 frames
  (about 23 minutes); stroke 151 trials, 30,130 frames. GenGait trained on 150
  people with nine trials each, so our normative set is smaller.
- The healthy subject names stored in the C3D files look like a birth date plus
  a sex letter. They are not used or written anywhere in this repository; the
  folder IDs (SUBJ01, TVC03) are used instead.
- Age, sex and lesion side are not in the C3D files, SubjectChar.xlsx or the
  post-processed Excel files. Per the dataset description they are only in the
  post-processed MAT files (sub_char).

Feature construction follows the paper: per joint, sin and cos of the three
Cardan angles plus the 6D rotation representation, 12 values. The rotation
matrix is built from the Plug-in Gait angles with an intrinsic Y-X-Z sequence
rather than from segment axes; this is an approximation to be checked.
