#!/bin/bash
# Irregular / asynchronous time-series benchmarks (see documents/irregular_benchmarks.md).
#
#   data/irregular/tpatchgnn/{physionet,activity,ushcn}   t-PatchGNN (ICML 2024) processed quartet, minus MIMIC
#   data/irregular/physiome_ode/                          Physiome-ODE (50 ODE-derived IMTS datasets), Zenodo 11492058
#   data/irregular/easytpp/{taxi,stackoverflow,retweet}   EasyTPP event sequences (timestamp + type), HF hub
#
# MIMIC-III-Ext-tPatchGNN needs the PhysioNet credentialed licence; fetch it by hand from
# https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/ into data/irregular/tpatchgnn/mimic/.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/data/irregular"
mkdir -p "$D"

# --- t-PatchGNN quartet (shipped in the repo, ~120 MB) ---
if [ ! -f "$D/tpatchgnn/ushcn/processed/ushcn.pt" ]; then
    echo "=== t-PatchGNN datasets ==="
    TMP=$(mktemp -d)
    git clone -q --depth 1 https://github.com/usail-hkust/t-PatchGNN "$TMP/tpatchgnn"
    mkdir -p "$D/tpatchgnn/physionet"
    cp -R "$TMP/tpatchgnn/data/activity" "$TMP/tpatchgnn/data/ushcn" "$D/tpatchgnn/"
    cp "$TMP/tpatchgnn/data/physionet/"*.zip "$D/tpatchgnn/physionet/"
    (cd "$D/tpatchgnn/physionet" && unzip -qo processed.zip && rm processed.zip)
    rm -rf "$TMP"
fi

# --- Physiome-ODE (715 MB zip) ---
if [ ! -d "$D/physiome_ode" ] || [ -z "$(ls -A "$D/physiome_ode" 2>/dev/null)" ]; then
    echo "=== Physiome-ODE ==="
    mkdir -p "$D/physiome_ode"
    curl -L --fail -o "$D/physiome_ode/final.zip" \
        "https://zenodo.org/api/records/11492058/files/final.zip/content"
    (cd "$D/physiome_ode" && unzip -qo final.zip && rm final.zip)
fi

# --- EasyTPP event sequences (a few MB each) ---
for ds in taxi stackoverflow retweet; do
    if [ ! -f "$D/easytpp/$ds/train.json" ]; then
        echo "=== EasyTPP $ds ==="
        mkdir -p "$D/easytpp/$ds"
        for f in train dev test; do
            curl -sL --fail -o "$D/easytpp/$ds/$f.json" \
                "https://huggingface.co/datasets/easytpp/$ds/resolve/main/$f.json"
        done
    fi
done

# --- Time-IMM (NeurIPS 2025 D&B; eight redistributable datasets, MIMIC excluded) ---
if [ ! -d "$D/timeimm/data/GDELT/processed" ]; then
    echo "=== Time-IMM ==="
    git clone -q --depth 1 https://github.com/blacksnail789521/Time-IMM "$D/timeimm"
fi

# --- Pyrregular (ICLR 2026; 34 standardized irregular datasets, HF hub) ---
# The many-entity sets this project cares about.  Files are HDF5, a few MB to
# a few hundred MB each; downloaded alone so the 3.9 GB collection is not
# pulled wholesale.  Licenses per dataset are in the pyrregular metadata.
PYR_HF="https://huggingface.co/datasets/splandi/pyrregular/resolve/data-v1/data_final"
PYR_DATASETS="Taxi GeolifeSupervised Vehicles Ldfpa Mimic3 Physionet2012 Physionet2019 Seabirds Pamap2 InsectWingbeat"
for ds in $PYR_DATASETS; do
    if [ ! -f "$D/pyrregular/$ds.h5" ]; then
        echo "=== Pyrregular $ds ==="
        mkdir -p "$D/pyrregular"
        curl -sL --fail -o "$D/pyrregular/$ds.h5" "$PYR_HF/$ds.h5"
    fi
done

du -sh "$D"/*
