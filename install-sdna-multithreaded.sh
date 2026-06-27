#!/usr/bin/env bash
# install-sdna-multithreaded.sh
#
# Rebuilds the sDNA+ shared library (sdna_vs2008.so) from source with
# OpenMP multithreading enabled. The pre-built wheel ships a .so compiled
# without OpenMP, so sDNA runs single-threaded even on multi-core machines.
#
# This script:
#   1. Locates the pre-installed sDNA+ package (from sdna-plus submodule)
#   2. Compiles all source files with g++ -fopenmp -O2
#   3. Replaces the pre-built .so with the OpenMP-enabled one
#   4. Verifies libgomp is linked
#
# Requires: g++, make, ld, dlopen-compatible libc (any Linux)
# Does NOT need: vcpkg, CMake, GEOS dev headers (GEOS loaded at runtime)
#
# Usage: bash install-sdna-multithreaded.sh
#   Or:  OMP_NUM_THREADS=8 bash install-sdna-multithreaded.sh  (set thread count)

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SDNA_SRC="${REPO_ROOT}/sdna-plus"
VENV_DIR="${REPO_ROOT}/.venv"
OMP_FLAGS="-fopenmp"
OMP_LINK="-lgomp"
BUILD_DIR=$(mktemp -d -t sdna_build_XXXXXX)
NCORES=$(nproc)

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}→${NC} $1"; }
warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
err()   { echo -e "${RED}✖${NC} $1"; }

cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

# ── Checks ──────────────────────────────────────────────────────────
if [ ! -d "$SDNA_SRC/sDNA/sdna_vs2008" ]; then
    err "sDNA+ source not found at ${SDNA_SRC}"
    err "Expected: ${SDNA_SRC}/sDNA/sdna_vs2008/"
    err "Make sure you've cloned the sdna-plus submodule:"
    err "  git submodule update --init sdna-plus"
    exit 1
fi

if ! command -v g++ &>/dev/null; then
    err "g++ not found. Install build-essential: sudo apt install build-essential"
    exit 1
fi

# Find the installed package directory
PACKAGE_DIR=$(python3 -c "
import sdna_plus, os
p = os.path.dirname(sdna_plus.__file__)
print(p)
" 2>/dev/null || true)

if [ -z "$PACKAGE_DIR" ]; then
    # Try .venv
    PACKAGE_DIR="${VENV_DIR}/lib/python3.13/site-packages/sDNA"
    if [ ! -d "$PACKAGE_DIR" ]; then
        # Search more broadly
        PACKAGE_DIR=$(find "${VENV_DIR}/lib" -path '*/site-packages/sDNA' -type d 2>/dev/null | head -1 || true)
    fi
fi

if [ ! -d "$PACKAGE_DIR/x64" ]; then
    err "sDNA+ package not found. Install first:"
    err "  uv pip install sdna-plus  (or pip install sdna-plus)"
    err "  Searched: ${PACKAGE_DIR}"
    exit 1
fi

info "sDNA+ source: ${SDNA_SRC}"
info "Package dir: ${PACKAGE_DIR}"
info "Build dir:   ${BUILD_DIR}"
info "Cores:       ${NCORES}"

# ── Source files ────────────────────────────────────────────────────
SDNA_CPP=()
for f in "$SDNA_SRC/sDNA/sdna_vs2008"/*.cpp; do
    SDNA_CPP+=("$f")
done
MUPARSER_CPP=()
for f in "$SDNA_SRC/sDNA/muparser/drop/src"/*.cpp; do
    MUPARSER_CPP+=("$f")
done

info "Found ${#SDNA_CPP[@]} sDNA source files"
info "Found ${#MUPARSER_CPP[@]} MuParser source files"

# ── Compile flags ───────────────────────────────────────────────────
CXXFLAGS="-std=c++14 -O2 -fPIC ${OMP_FLAGS} -DNDEBUG"
CXXFLAGS+=" -I${SDNA_SRC}/sDNA/sdna_vs2008"
CXXFLAGS+=" -I${SDNA_SRC}/sDNA/muparser/drop/include"
CXXFLAGS+=" -I${SDNA_SRC}/sDNA/sdna_vs2008"
CXXFLAGS+=" -I${SDNA_SRC}/sDNA/sdna_vs2008/.."  # for parent includes if any

LDFLAGS="-shared ${OMP_LINK} -ldl -lpthread"

# ── Compile objects ─────────────────────────────────────────────────
info "Compiling sDNA source files..."
OBJS=()
for src in "${SDNA_CPP[@]}" "${MUPARSER_CPP[@]}"; do
    basename=$(basename "$src" .cpp)
    obj="${BUILD_DIR}/${basename}.o"
    info "  cc -O2 ${basename}.cpp"
    g++ $CXXFLAGS -c "$src" -o "$obj"
    OBJS+=("$obj")
done

# ── Link ─────────────────────────────────────────────────────────────
info "Linking sdna_vs2008.so..."
g++ "${OBJS[@]}" $LDFLAGS -o "${BUILD_DIR}/sdna_vs2008.so"

# ── Check for OpenMP ────────────────────────────────────────────────
if readelf -d "${BUILD_DIR}/sdna_vs2008.so" 2>/dev/null | grep -q 'NEEDED.*gomp'; then
    info "${GREEN}✓${NC} OpenMP (libgomp) linked — multi-threading enabled"
else
    warn "libgomp not found in NEEDED section. OpenMP may not be active."
    warn "Checking symbols..."
    if nm -D "${BUILD_DIR}/sdna_vs2008.so" 2>/dev/null | grep -q 'GOMP_parallel\|omp_get'; then
        info "${GREEN}✓${NC} OpenMP symbols found (statically linked)"
    else
        warn "No OpenMP symbols found. Falling back to pre-built .so"
        rm -f "${BUILD_DIR}/sdna_vs2008.so"
        exit 1
    fi
fi

SO_SIZE=$(stat -c%s "${BUILD_DIR}/sdna_vs2008.so" 2>/dev/null || echo 0)
info "${GREEN}✓${NC} Built: ${BUILD_DIR}/sdna_vs2008.so ($((SO_SIZE / 1024)) KB)"

# ── Back up old .so and install new one ──────────────────────────────
OLD_SO="${PACKAGE_DIR}/x64/sdna_vs2008.so"
if [ -f "$OLD_SO" ]; then
    OLD_SIZE=$(stat -c%s "$OLD_SO" 2>/dev/null || echo 0)
    BACKUP="${OLD_SO}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$OLD_SO" "$BACKUP"
    info "Backed up old .so to ${BACKUP} ($((OLD_SIZE / 1024)) KB)"
fi

cp "${BUILD_DIR}/sdna_vs2008.so" "$OLD_SO"
info "${GREEN}✓${NC} Installed to ${OLD_SO}"

# ── Also copy libgeos_c.so if needed ─────────────────────────────────
# The .so looks for libgeos_c.so in the same directory as itself.
SYS_GEOS=$(find /usr/lib/x86_64-linux-gnu -name 'libgeos_c.so*' 2>/dev/null | head -1 || true)
if [ -n "$SYS_GEOS" ]; then
    cp "$SYS_GEOS" "${PACKAGE_DIR}/x64/libgeos_c.so" 2>/dev/null || true
    if [ -f "${PACKAGE_DIR}/x64/libgeos_c.so" ]; then
        info "${GREEN}✓${NC} Copied ${SYS_GEOS} → ${PACKAGE_DIR}/x64/"
    fi
fi

# ── Quick verification ──────────────────────────────────────────────
info ""
info "── Verification ──"
info "Testing sDNA+ import..."
python3 -c "
import sdna_plus
import os
so_path = os.path.join(os.path.dirname(sdna_plus.__file__), 'x64', 'sdna_vs2008.so')
if os.path.exists(so_path):
    import subprocess
    r = subprocess.run(['readelf', '-d', so_path], capture_output=True, text=True)
    if 'gomp' in r.stdout:
        print('  ✓ OpenMP (libgomp) linked in .so')
    else:
        print('  ⚠ No libgomp found — check compilation flags')
    r2 = subprocess.run(['nm', '-D', so_path], capture_output=True, text=True)
    omp_count = sum(1 for line in r2.stdout.split('\\n') if 'GOMP' in line or 'omp_get' in line)
    print(f'  ✓ {omp_count} OpenMP symbols in .so')
" 2>&1

info ""
info "${GREEN}── Installation complete ──${NC}"
info "Thread count is controlled by OMP_NUM_THREADS (default: all cores)"
info "Run your sDNA benchmark: .venv/bin/python scripts/bench_sdna.py"
info ""
info "To verify multi-core usage, watch htop while sDNA runs."
