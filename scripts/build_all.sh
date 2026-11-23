#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"

LINUX_TARGET="x86_64-unknown-linux-gnu"
WIN_TARGET="x86_64-pc-windows-gnu"

LINUX_SO="${ROOT_DIR}/target/${LINUX_TARGET}/release/libOmniMIDI.so"
WIN_DLL="${ROOT_DIR}/target/${WIN_TARGET}/release/OmniMIDI.dll"
LINUX_GUI="${ROOT_DIR}/target/${LINUX_TARGET}/release/kdmapiv3-config"
WIN_GUI="${ROOT_DIR}/target/${WIN_TARGET}/release/kdmapiv3-config.exe"

REQUIRED_SYMBOLS=(
    DriverSettings
    GetRenderingTime
    GetVoiceCount
    InitializeCallbackFeatures
    InitializeKDMAPIStream
    IsKDMAPIAvailable
    LoadCustomSoundFontsList
    PrepareLongData
    ResetKDMAPIStream
    ReturnKDMAPIVer
    RunCallbackFunction
    SendCustomEvent
    SendDirectData
    SendDirectDataNoBuf
    SendDirectLongData
    SendDirectLongDataNoBuf
    TerminateKDMAPIStream
    UnprepareLongData
)

echo "==> [1/6] Building Linux native cdylib (${LINUX_TARGET})..."
(cd "${ROOT_DIR}" && cargo build --release -p kdmapiv3 --target "${LINUX_TARGET}")

echo "==> [2/6] Building Windows/Wine PE cdylib (${WIN_TARGET})..."
(cd "${ROOT_DIR}" && cargo build --release -p kdmapiv3 --target "${WIN_TARGET}")

echo "==> [3/6] Building Linux GUI configurator (kdmapiv3-config)..."
(cd "${ROOT_DIR}" && cargo build --release -p kdmapiv3-config --target "${LINUX_TARGET}")

echo "==> [4/6] Building Windows/Wine GUI configurator (kdmapiv3-config.exe)..."
(cd "${ROOT_DIR}" && cargo build --release -p kdmapiv3-config --target "${WIN_TARGET}")

echo "==> [5/6] Verifying exported KDMAPI symbols..."

LINUX_EXPORTS="$(nm -D --defined-only "${LINUX_SO}" | awk '{print $3}' | sort)"
for sym in "${REQUIRED_SYMBOLS[@]}"; do
    if ! grep -qx "${sym}" <<< "${LINUX_EXPORTS}"; then
        echo "ERROR: Missing expected Linux symbol in libOmniMIDI.so: ${sym}" >&2
        exit 1
    fi
done
LINUX_COUNT="$(wc -l <<< "${LINUX_EXPORTS}")"
if [[ "${LINUX_COUNT}" -ne "${#REQUIRED_SYMBOLS[@]}" ]]; then
    echo "ERROR: Expected exactly ${#REQUIRED_SYMBOLS[@]} Linux exports, found ${LINUX_COUNT}" >&2
    echo "${LINUX_EXPORTS}" >&2
    exit 1
fi
echo "    [OK] libOmniMIDI.so exports exactly ${LINUX_COUNT} clean KDMAPI symbols."

OBJDUMP_WIN="x86_64-w64-mingw32-objdump"
if ! command -v "${OBJDUMP_WIN}" >/dev/null 2>&1; then
    OBJDUMP_WIN="objdump"
fi

WIN_EXPORTS="$("${OBJDUMP_WIN}" -p "${WIN_DLL}" \
    | sed -n '/\[Ordinal\/Name Pointer\] Table/,/^$/p' \
    | awk '/^[[:space:]]*\[[[:space:]]*[0-9]+\]/ {print $NF}' \
    | sort)"
for sym in "${REQUIRED_SYMBOLS[@]}"; do
    if ! grep -qx "${sym}" <<< "${WIN_EXPORTS}"; then
        echo "ERROR: Missing expected Windows DLL export in OmniMIDI.dll: ${sym}" >&2
        exit 1
    fi
done
WIN_COUNT="$(wc -l <<< "${WIN_EXPORTS}")"
if [[ "${WIN_COUNT}" -ne "${#REQUIRED_SYMBOLS[@]}" ]]; then
    echo "ERROR: Expected exactly ${#REQUIRED_SYMBOLS[@]} Windows exports, found ${WIN_COUNT}" >&2
    echo "${WIN_EXPORTS}" >&2
    exit 1
fi
echo "    [OK] OmniMIDI.dll exports exactly ${WIN_COUNT} clean undecorated KDMAPI symbols."

echo "==> [6/6] Packaging drop-in bundle into dist/..."
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/linux" "${DIST_DIR}/windows"

cp "${LINUX_SO}" "${DIST_DIR}/linux/libOmniMIDI.so"
cp "${LINUX_GUI}" "${DIST_DIR}/linux/kdmapiv3-config"
cp "${WIN_DLL}" "${DIST_DIR}/windows/OmniMIDI.dll"
cp "${WIN_GUI}" "${DIST_DIR}/windows/kdmapiv3-config.exe"

cat > "${DIST_DIR}/kdmapiv3.toml" <<'EOF'
# kdmapiv3 configuration file
# Loaded once during InitializeKDMAPIStream (never touched on the audio/MIDI hot path).
#
# Placement priority:
#   1. $KDMAPIV3_CONFIG (optional environment variable override)
#   2. Next to libOmniMIDI.so / OmniMIDI.dll (portable drop-in next to Wasabi or PFA)
#   3. Linux:   ~/.config/kdmapiv3/kdmapiv3.toml
#      Windows: %APPDATA%\kdmapiv3\kdmapiv3.toml

# Maximum simultaneous voices for Black MIDI playback (1 - 1,000,000; default: 10000).
voice_limit = 10000

# Resampling interpolation quality: "linear", "sinc8", "sinc16", or "sinc32" (default: "linear").
interpolation = "linear"

# Audio render buffer size in frames (32 - 8192; default: 512).
buffer_frames = 512

# Total DSP render threads (1 = single-threaded, 2 - 256 = multi-threaded; default: 4).
render_threads = 4

# Pre-allocated lock-free MIDI ring buffer slots (power of two, 1024 - 16777216; default: 1048576).
ring_buffer_capacity = 1048576

# Ordered list of SoundFont (.sf2 / .sfz) paths.
# Windows backslashes, Wine Z:\ paths, and mismatched casing are normalized automatically.
# Example:
# soundfonts = [
#     '/home/user/SoundFonts/KeppyGrand.sf2',
#     'C:\SoundFonts\ExtraPercussion.sf2',
# ]
soundfonts = []

# Optional non-blocking diagnostics: when true, a separate low-priority background thread
# writes a 1-second atomic counter summary to kdmapiv3.log without touching the hot path.
telemetry_enabled = false

# Low-Spec / Black MIDI DSP Effect Toggles (zero hot-path overhead when disabled).
[effects]
enable_lowpass_filter = true
enable_vibrato_lfo = true
disable_all_effects = false
EOF

cp "${DIST_DIR}/kdmapiv3.toml" "${DIST_DIR}/linux/kdmapiv3.toml"
cp "${DIST_DIR}/kdmapiv3.toml" "${DIST_DIR}/windows/kdmapiv3.toml"

echo ""
echo "Done! Drop-in artifacts ready in: ${DIST_DIR}"
ls -lh \
    "${DIST_DIR}/linux/libOmniMIDI.so" \
    "${DIST_DIR}/linux/kdmapiv3-config" \
    "${DIST_DIR}/windows/OmniMIDI.dll" \
    "${DIST_DIR}/windows/kdmapiv3-config.exe" \
    "${DIST_DIR}/kdmapiv3.toml"
