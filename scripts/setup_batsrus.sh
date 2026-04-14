#!/usr/bin/env bash
# Build BATSRUS as a self-contained artifact inside MIDL-Pipeline/BATSRUS/.
# Idempotent: re-running skips completed steps.
#
# The build recipe here is the exact sequence that works on the UMich csem
# cluster (Linux, gfortran, openmpi via /usr/lib64/openmpi/bin). Flags are
# documented inline.
#
# Usage:
#     bash MIDL-Pipeline/scripts/setup_batsrus.sh
#
# Result:
#     MIDL-Pipeline/BATSRUS/run_mhd/BATSRUS.exe   -- ready-to-run template
#     MIDL-Pipeline/BATSRUS/run_mhd/PARAM.in.MIDL -- MIDL PARAM template

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BATSRUS_DIR="$PIPELINE_DIR/BATSRUS"
RUN_DIR_NAME="run_mhd"
TEMPLATE_SRC="$SCRIPT_DIR/PARAM.in.MIDL"

# Put openmpi on PATH for Config.pl and make
export PATH="/usr/lib64/openmpi/bin:${PATH}"

echo "[setup_batsrus] PIPELINE_DIR = $PIPELINE_DIR"
echo "[setup_batsrus] BATSRUS_DIR  = $BATSRUS_DIR"

# ---------------------------------------------------------------------------
# Step 1: clone BATSRUS (HTTPS — SSH auth is not configured on this cluster)
# ---------------------------------------------------------------------------
if [ ! -d "$BATSRUS_DIR/.git" ]; then
    echo "[setup_batsrus] Step 1: cloning BATSRUS via HTTPS..."
    git clone https://github.com/SWMFsoftware/BATSRUS.git "$BATSRUS_DIR"
else
    echo "[setup_batsrus] Step 1: BATSRUS already cloned, skipping."
fi

cd "$BATSRUS_DIR"

# ---------------------------------------------------------------------------
# Step 2: clone srcBATL manually via HTTPS.
# Config.pl -install tries to git clone srcBATL via SSH, which fails on this
# cluster because there is no SSH key. We pre-clone via HTTPS so Config.pl
# finds it already in place and does not try.
# ---------------------------------------------------------------------------
if [ ! -d "$BATSRUS_DIR/srcBATL/.git" ] && [ ! -f "$BATSRUS_DIR/srcBATL/Makefile" ]; then
    echo "[setup_batsrus] Step 2: cloning srcBATL via HTTPS..."
    rm -rf "$BATSRUS_DIR/srcBATL"
    git clone https://github.com/SWMFsoftware/srcBATL.git "$BATSRUS_DIR/srcBATL"
else
    echo "[setup_batsrus] Step 2: srcBATL already present, skipping."
fi

# ---------------------------------------------------------------------------
# Step 3: Config.pl -install with gfortran + mpif90 + gcc (no SSH)
# ---------------------------------------------------------------------------
if [ ! -f "$BATSRUS_DIR/Makefile.conf" ]; then
    echo "[setup_batsrus] Step 3: running Config.pl -install..."
    ./Config.pl -install -compiler=mpif90,gcc_mpicc
else
    echo "[setup_batsrus] Step 3: Makefile.conf already exists, skipping -install."
fi

# ---------------------------------------------------------------------------
# Step 4: patch Makefile.conf for gfortran-compatible flags.
#  - DOUBLEPREC needs -frecord-marker=4 so unformatted binary reads match
#    the default spacepy/CDF sizes expected by BATSRUS share routines.
#  - CFLAG needs -cpp (enable preprocessor) and -ffree-line-length-none
#    (some share/*.f90 lines exceed 132 cols and gfortran chokes otherwise).
# ---------------------------------------------------------------------------
echo "[setup_batsrus] Step 4: patching Makefile.conf for gfortran..."
python3 - "$BATSRUS_DIR/Makefile.conf" <<'PYEOF'
import re, sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()

want_dp = 'DOUBLEPREC = -fdefault-real-8 -fdefault-double-8 -frecord-marker=4'
text = re.sub(r'^DOUBLEPREC\s*=.*$', want_dp, text, count=1, flags=re.MULTILINE)

want_cflag = 'CFLAG = ${SEARCH} -c -w -cpp -ffree-line-length-none ${DEBUG}'
text = re.sub(r'^CFLAG\s*=.*$', want_cflag, text, count=1, flags=re.MULTILINE)

with open(path, 'w') as f:
    f.write(text)
print(f"  patched {path}")
PYEOF

# ---------------------------------------------------------------------------
# Step 5: configure for 1D MHD, 10x1x1 block with 3 ghost cells (25 root
#   blocks * 10 cells = 250 cells along X at native 1.0 Re/cell).
#
# Cells-per-block is baked into Fortran sources at configure time, so we
# always `make clean` here to force a rebuild if a previous configuration
# used a different block size. This is required for idempotency when
# switching between -g=8 and -g=10 builds.
# ---------------------------------------------------------------------------
echo "[setup_batsrus] Step 5: make clean (force rebuild for new block size)..."
make clean

echo "[setup_batsrus] Step 5: Config.pl -default -e=Mhd -u=Default -g=10,1,1 -ng=3..."
./Config.pl -default -e=Mhd -u=Default -g=10,1,1 -ng=3

# ---------------------------------------------------------------------------
# Step 6: build BATSRUS.exe (always run after clean above).
# ---------------------------------------------------------------------------
echo "[setup_batsrus] Step 6: make -j8 BATSRUS..."
make -j8 BATSRUS

# ---------------------------------------------------------------------------
# Step 7: create the reusable run directory template
# ---------------------------------------------------------------------------
if [ ! -d "$BATSRUS_DIR/$RUN_DIR_NAME" ]; then
    echo "[setup_batsrus] Step 7: make test_rundir TESTDIR=$RUN_DIR_NAME..."
    make test_rundir TESTDIR="$RUN_DIR_NAME"
else
    echo "[setup_batsrus] Step 7: $RUN_DIR_NAME already exists, skipping."
fi

# ---------------------------------------------------------------------------
# Step 8: drop the MIDL PARAM template into the run directory
# ---------------------------------------------------------------------------
echo "[setup_batsrus] Step 8: copying PARAM.in.MIDL template into $RUN_DIR_NAME/..."
cp "$TEMPLATE_SRC" "$BATSRUS_DIR/$RUN_DIR_NAME/PARAM.in.MIDL"

# ---------------------------------------------------------------------------
# Step 9: verify by running the imf19980504 sample case through run_mhd/.
#   We stage PARAM.in.L1toBC + imf19980504.dat (as L1.dat) and run BATSRUS.
# ---------------------------------------------------------------------------
RUN_DIR="$BATSRUS_DIR/$RUN_DIR_NAME"
LOG_OUT="$RUN_DIR/GM/IO2/log_n000000.log"

# Always re-verify the sample case after a rebuild, since the cells-per-block
# change above forces a fresh BATSRUS.exe. Clear any stale output first.
echo "[setup_batsrus] Step 9: running imf19980504 verification case..."
cd "$RUN_DIR"
rm -f GM/IO2/log_n000000.log GM/IO2/*.idl runlog
cp "$BATSRUS_DIR/Param/EARTH/PARAM.in.L1toBC" PARAM.in
cp "$BATSRUS_DIR/Param/EARTH/imf19980504.dat" L1.dat
ulimit -s unlimited || true
./BATSRUS.exe > runlog 2>&1
cd "$BATSRUS_DIR"

echo ""
echo "[setup_batsrus] Build complete."
echo "[setup_batsrus] BATSRUS.exe: $RUN_DIR/BATSRUS.exe"
echo "[setup_batsrus] Verification log (last 3 lines):"
tail -n 3 "$LOG_OUT" || echo "  (no log yet)"
