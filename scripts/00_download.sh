#!/usr/bin/env bash
# Download the public gait datasets into data/raw/ and check md5 sums.
# Re-running skips files that are already present with the right checksum.
set -euo pipefail
cd "$(dirname "$0")/.."

md5of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

fetch() {  # url dest md5
  local url=$1 dest=$2 sum=$3
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" && "$(md5of "$dest")" == "$sum" ]]; then
    echo "ok (present) $dest"; return
  fi
  curl -L --fail --retry 3 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  if [[ "$(md5of "$dest")" != "$sum" ]]; then
    echo "md5 mismatch: $dest" >&2; exit 1
  fi
  echo "ok $dest"
}

# Van Criekinge et al. 2023, Sci Data 10:852. Figshare collection 10.6084/m9.figshare.c.6503791.v1
VC=data/raw/vancriekinge2023
fetch https://ndownloader.figshare.com/files/42450060 $VC/138_HealthyPiG.zip 85235c73455d4df8434810d61517efbb
fetch https://ndownloader.figshare.com/files/42450069 $VC/50_StrokePiG.zip 71925450f3c058b145791de7951c8dc9
fetch https://ndownloader.figshare.com/files/42450090 $VC/NrStrides.xlsx d66c33d1b460dfd7dec38767453b484e
fetch https://ndownloader.figshare.com/files/42450093 $VC/MATdatafiles_description_v1.3_LST.xlsx 91e19f0c0bbac2cabab087bf17cbbc65
# Post-processed Excel files: needed for age, sex, lesion side and clinical scores
fetch https://ndownloader.figshare.com/files/42450072 $VC/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx 7d6585bb1bfd61f53840aa3a0eb288de
fetch https://ndownloader.figshare.com/files/42450084 $VC/MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx d79ec047d50f5a957d164a25bffabf32

# Shida et al. 2023, Front Neurosci 17:992585. Figshare 10.6084/m9.figshare.14896881.v4
SH=data/raw/shida2023
fetch https://ndownloader.figshare.com/files/28739484 $SH/C3Dfiles.zip a3f2989cbcd2873738d1c9050d0296ee
fetch https://ndownloader.figshare.com/files/37907001 $SH/PDGinfo.xlsx 532dfce5e7150b4946b7c3cd7a443363
fetch https://ndownloader.figshare.com/files/38431859 $SH/Supplementary_material.docx b9790caa1210f067ac330cf69b738d5a
