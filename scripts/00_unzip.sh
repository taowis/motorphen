#!/usr/bin/env bash
# Unzip the downloaded C3D archives into data/raw/*/extracted/.
set -euo pipefail
cd "$(dirname "$0")/.."
VC=data/raw/vancriekinge2023
unzip -q -o $VC/138_HealthyPiG.zip -d $VC/extracted
unzip -q -o $VC/50_StrokePiG.zip -d $VC/extracted
# one stroke subject is shipped as a nested zip
for z in $VC/extracted/50_StrokePiG/*.zip; do unzip -q -o "$z" -d $VC/extracted/50_StrokePiG; done
unzip -q -o data/raw/shida2023/C3Dfiles.zip -d data/raw/shida2023/extracted
echo done
