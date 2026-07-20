#!/usr/bin/env bash
#
# Bootstrap the retrieval toolkit: virtualenv, dependencies, index, manifest,
# and a live check that the MCP server answers a real handshake.
#
# Invocable from anywhere. Every path is derived from this script's own
# location, never from $PWD:
#
#     ./rag/install.sh                 # from the workspace root
#     ./install.sh                     # from inside rag/
#     /abs/path/to/rag/install.sh      # from anywhere at all
#
# Idempotent: re-running reuses a working virtualenv, resets the target
# collection before rebuilding it so no orphaned chunks survive, and rewrites
# the manifest only when the body actually changed.
#
# Offline-tolerant: dependency installation is attempted with the index
# disabled first and only falls back to the network if that is not enough. The
# health report states which of the two happened.
#
#     ./install.sh --profile chunk-small
#     ./install.sh --recreate            # rebuild the virtualenv from scratch
#     ./install.sh --wheelhouse ./wheels # offline install from local wheels
#     ./install.sh --refresh             # after a KB edit: reindex + manifest
#
# A missing knowledge base is not a failure of a full install. A fresh clone of
# the public repository has no KB/ — the notes are the one thing that is never
# published — so the index and manifest steps report SKIP and the run still
# exits 0 with a usable virtualenv. The benchmarks need no KB/ at all: they
# ship their own corpus.
#
#     ./install.sh && python bench.py --all --corpus bench/corpus
#
# --refresh is stricter: reindexing after an edit with nothing to reindex means
# the body is not where this script is looking, and that still FAILs.
#
# --refresh is the daily-loop form: it runs the profile check, the index step
# and the manifest step, and nothing else. It never creates or touches the
# virtualenv, never runs pip, and never opens the MCP handshake — if the venv
# is missing or broken it fails fast and tells you to run the full install.
# The health report is the same, with the skipped steps shown as SKIP. The
# resulting manifest root is logged to logs/rag.log on every refresh, whether
# or not anything changed.
#
# Exit status is 0 only if every step passed; the health report names the step
# that failed.

set -u -o pipefail

# --- locate ourselves -------------------------------------------------------

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    LINK_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    case "$SOURCE" in
        /*) ;;
        *) SOURCE="$LINK_DIR/$SOURCE" ;;
    esac
done
RAG_ROOT="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -P "$RAG_ROOT/.." >/dev/null 2>&1 && pwd)"

VENV_DIR="$RAG_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
REQUIREMENTS="$RAG_ROOT/requirements.txt"
KB_DIR="${RAG_KB_ROOT:-$WORKSPACE_ROOT/KB}"

# --- arguments --------------------------------------------------------------

PROFILE="baseline"
RECREATE=0
REFRESH=0
WHEELHOUSE=""

usage() {
    # The header block itself is the help text; print it until the comments
    # stop, so growing the header never silently truncates --help.
    awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^#[ ]?/, ""); print }' "$SOURCE"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --profile)
            [ $# -ge 2 ] || { echo "--profile needs a name" >&2; exit 2; }
            PROFILE="$2"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        --recreate) RECREATE=1; shift ;;
        --refresh) REFRESH=1; shift ;;
        --wheelhouse)
            [ $# -ge 2 ] || { echo "--wheelhouse needs a directory" >&2; exit 2; }
            WHEELHOUSE="$2"; shift 2 ;;
        --wheelhouse=*) WHEELHOUSE="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

PROFILE_FILE="$RAG_ROOT/profiles/$PROFILE.json"

if [ "$REFRESH" = "1" ] && [ "$RECREATE" = "1" ]; then
    echo "--refresh and --recreate are opposites: refresh never touches the venv" >&2
    exit 2
fi

# --- health report bookkeeping ---------------------------------------------

STEP_LINES=()
FAILED_STEP=""
STEP_START=0

step_begin() { STEP_START=$(date +%s); }

step_end() {
    # step_end <name> <status> <detail>
    local name="$1" status="$2" detail="$3"
    local elapsed=$(( $(date +%s) - STEP_START ))
    STEP_LINES+=("$(printf '%-14s %-6s %4ss  %s' "$name" "$status" "$elapsed" "$detail")")
    if [ "$status" = "FAIL" ] && [ -z "$FAILED_STEP" ]; then
        FAILED_STEP="$name"
    fi
}

report_and_exit() {
    echo
    echo "==================== rag install health report ===================="
    printf '%-14s %-6s %5s  %s\n' "step" "status" "time" "detail"
    echo "-------------------------------------------------------------------"
    local line
    for line in "${STEP_LINES[@]}"; do
        echo "$line"
    done
    echo "-------------------------------------------------------------------"
    printf '%-14s %s\n' "rag root" "$RAG_ROOT"
    printf '%-14s %s\n' "knowledge" "$KB_DIR$([ -d "$KB_DIR" ] || echo '  (absent - index and manifest skipped)')"
    printf '%-14s %s\n' "profile" "$PROFILE ($PROFILE_FILE)"
    printf '%-14s %s\n' "mode" "$([ "$REFRESH" = "1" ] && echo 'refresh (reindex + manifest only)' || echo 'full install')"
    printf '%-14s %s\n' "invoked from" "$PWD"
    if [ -n "$FAILED_STEP" ]; then
        echo
        echo "RESULT: FAILED at step '$FAILED_STEP' - see the detail column above."
        echo "==================================================================="
        exit 1
    fi
    echo
    if [ "$REFRESH" = "1" ]; then
        echo "RESULT: OK - index and manifest refreshed."
    else
        echo "RESULT: OK - toolkit installed and answering."
    fi
    echo "==================================================================="
    exit 0
}

step_skip() {
    # step_skip <name> <detail> — a step deliberately not run (--refresh).
    STEP_LINES+=("$(printf '%-14s %-6s %4ss  %s' "$1" "SKIP" "0" "$2")")
}

skip_rest() {
    # Mark every remaining step as skipped once something has failed.
    local name
    for name in "$@"; do
        STEP_LINES+=("$(printf '%-14s %-6s %4ss  %s' "$name" "SKIP" "0" "skipped after earlier failure")")
    done
    report_and_exit
}

# --- step 1: profile --------------------------------------------------------

step_begin
if [ ! -f "$PROFILE_FILE" ]; then
    AVAILABLE="$(ls "$RAG_ROOT/profiles"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.json$//' | tr '\n' ' ')"
    step_end "profile" "FAIL" "no such profile '$PROFILE'; have: ${AVAILABLE:-none}"
    skip_rest "venv" "deps" "index" "manifest" "mcp"
fi
PROFILE_COLLECTION="$(sed -n 's/.*"collection"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$PROFILE_FILE" | head -1)"
PROFILE_KIND="$(sed -n 's/.*"kind"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$PROFILE_FILE" | head -1)"
if [ "$PROFILE_KIND" != "local" ]; then
    step_end "profile" "FAIL" "kind '$PROFILE_KIND' is reserved, not implemented"
    skip_rest "venv" "deps" "index" "manifest" "mcp"
fi
# A profile may declare a source and leave it switched off. Say so here rather
# than after a virtualenv rebuild: activating a source is the owner's decision.
if grep -q '"enabled"[[:space:]]*:[[:space:]]*false' "$PROFILE_FILE"; then
    step_end "profile" "FAIL" "profile '$PROFILE' declares a source with enabled=false; set it to true in $PROFILE_FILE to activate"
    skip_rest "venv" "deps" "index" "manifest" "mcp"
fi
step_end "profile" "OK" "$PROFILE (kind=$PROFILE_KIND, collection=$PROFILE_COLLECTION)"

# --- steps 2-3: virtualenv and dependencies ---------------------------------
#
# --refresh owns neither: it is reindex + manifest and nothing else. It checks
# that the venv is usable and stops if it is not, but never creates one and
# never runs pip.

if [ "$REFRESH" = "1" ]; then
    step_begin
    if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import chromadb" >/dev/null 2>&1; then
        VENV_VERSION="$("$VENV_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"
        CHROMA_VERSION="$("$VENV_PY" -c 'import chromadb; print(chromadb.__version__)' 2>/dev/null)"
        step_skip "venv" "refresh: not touched; existing .venv (python $VENV_VERSION)"
        step_skip "deps" "refresh: pip not run; chromadb $CHROMA_VERSION already importable"
    else
        step_end "venv" "FAIL" "no usable virtualenv at $VENV_DIR - run the full '$SOURCE' (without --refresh) first"
        skip_rest "deps" "index" "manifest" "mcp"
    fi
else

# --- step 2: virtualenv -----------------------------------------------------

step_begin
VENV_NOTE=""
if [ "$RECREATE" = "1" ] && [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
    VENV_NOTE="recreated"
fi

if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
    # Reuse. Creating a venv over a working one is what produces the "duplicate
    # layer" problem this script is meant not to have.
    VENV_VERSION="$("$VENV_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"
    step_end "venv" "OK" "${VENV_NOTE:-reused} existing .venv (python $VENV_VERSION)"
else
    BASE_PY=""
    for candidate in python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
                BASE_PY="$candidate"
                break
            fi
        fi
    done
    if [ -z "$BASE_PY" ]; then
        step_end "venv" "FAIL" "no python >= 3.9 on PATH"
        skip_rest "deps" "index" "manifest" "mcp"
    fi
    rm -rf "$VENV_DIR"
    mkdir -p "$RAG_ROOT/logs"
    VENV_LOG="$RAG_ROOT/logs/install-venv.log"
    if ! "$BASE_PY" -m venv "$VENV_DIR" > "$VENV_LOG" 2>&1; then
        step_end "venv" "FAIL" "$BASE_PY -m venv failed: $(tail -1 "$VENV_LOG")"
        skip_rest "deps" "index" "manifest" "mcp"
    fi
    VENV_VERSION="$("$VENV_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"
    step_end "venv" "OK" "created .venv with $BASE_PY (python $VENV_VERSION)"
fi

# --- step 3: dependencies ---------------------------------------------------

step_begin
if [ ! -f "$REQUIREMENTS" ]; then
    step_end "deps" "FAIL" "missing $REQUIREMENTS"
    skip_rest "index" "manifest" "mcp"
fi

PIP_LOG="$RAG_ROOT/logs/install-pip.log"
mkdir -p "$RAG_ROOT/logs"
: > "$PIP_LOG"

OFFLINE_ARGS=(--no-index)
if [ -n "$WHEELHOUSE" ]; then
    OFFLINE_ARGS+=(--find-links "$WHEELHOUSE")
elif [ -d "$RAG_ROOT/wheels" ]; then
    OFFLINE_ARGS+=(--find-links "$RAG_ROOT/wheels")
fi

DEPS_MODE=""
{
    echo "### offline attempt: pip install ${OFFLINE_ARGS[*]} -r $REQUIREMENTS"
} >> "$PIP_LOG"
if "$VENV_PY" -m pip install --disable-pip-version-check --no-input \
        "${OFFLINE_ARGS[@]}" -r "$REQUIREMENTS" >> "$PIP_LOG" 2>&1; then
    DEPS_MODE="offline"
else
    echo "### offline attempt failed, falling back to the network" >> "$PIP_LOG"
    if "$VENV_PY" -m pip install --disable-pip-version-check --no-input \
            -r "$REQUIREMENTS" >> "$PIP_LOG" 2>&1; then
        DEPS_MODE="online"
    else
        step_end "deps" "FAIL" "offline and online installs both failed; see $PIP_LOG"
        skip_rest "index" "manifest" "mcp"
    fi
fi

if ! "$VENV_PY" -c "import chromadb" >/dev/null 2>&1; then
    step_end "deps" "FAIL" "chromadb still not importable after a '$DEPS_MODE' install"
    skip_rest "index" "manifest" "mcp"
fi
CHROMA_VERSION="$("$VENV_PY" -c 'import chromadb; print(chromadb.__version__)' 2>/dev/null)"
PKG_COUNT="$(grep -c '^[A-Za-z]' "$REQUIREMENTS" 2>/dev/null || echo '?')"
if [ "$DEPS_MODE" = "offline" ]; then
    step_end "deps" "OK" "offline (no network used); $PKG_COUNT pins satisfied, chromadb $CHROMA_VERSION"
else
    step_end "deps" "OK" "online fallback (offline attempt was insufficient); chromadb $CHROMA_VERSION"
fi

fi  # end of the non-refresh venv/deps branch

# --- step 4: index ----------------------------------------------------------
#
# A missing knowledge base is two different situations and they must not share
# an exit code:
#
#   full install  A fresh clone of the public repository has no KB/ at all —
#                 that is the normal state of this repository, not a fault. The
#                 toolkit still installs: virtualenv, dependencies, and the
#                 benchmarks, which bring their own corpus and never read KB/.
#                 Index and manifest are SKIPped and the run exits 0.
#   --refresh     "reindex after editing a note" with nothing to reindex is a
#                 real error: the body the caller believes they edited is not
#                 where this script looks. It still FAILs, exactly as before.
#
# Nothing about the path taken when KB_DIR *does* exist changes.

KB_ABSENT=0
if [ ! -d "$KB_DIR" ]; then
    KB_ABSENT=1
fi

if [ "$KB_ABSENT" = "1" ] && [ "$REFRESH" = "1" ]; then
    step_begin
    step_end "index" "FAIL" "no knowledge base directory at $KB_DIR (--refresh has nothing to reindex)"
    skip_rest "manifest" "mcp"
fi

if [ "$KB_ABSENT" = "1" ]; then
    step_skip "index" "no knowledge base at $KB_DIR - nothing to index (set RAG_KB_ROOT to point at one)"
    step_skip "manifest" "no knowledge base to describe; the shipped manifest.json is left untouched"
    if [ -f "$RAG_ROOT/bench.py" ] && [ -d "$RAG_ROOT/bench/corpus" ]; then
        BENCH_HINT="benchmarks are self-contained: python bench.py --all --corpus bench/corpus"
    else
        BENCH_HINT="point RAG_KB_ROOT at a directory of markdown notes and re-run"
    fi
    step_skip "kb" "$BENCH_HINT"
fi

if [ "$KB_ABSENT" = "0" ]; then

step_begin
# --reset drops the collection before rebuilding: no duplicates, no orphans.
INDEX_OUT="$(RAG_LOG_QUIET=1 "$VENV_PY" "$RAG_ROOT/index_toolbox.py" --reset \
    --profile "$PROFILE" --root "$KB_DIR" 2>&1)"
INDEX_RC=$?
if [ $INDEX_RC -ne 0 ]; then
    step_end "index" "FAIL" "index_toolbox.py exited $INDEX_RC: $(echo "$INDEX_OUT" | tail -1)"
    skip_rest "manifest" "mcp"
fi
step_end "index" "OK" "$(echo "$INDEX_OUT" | tail -1)"

# --- step 5: manifest -------------------------------------------------------
#
# manifest.json is the *published* artefact: it is what the public skeleton
# ships, and its root is a pure function of the body under the default chunk
# geometry. Whether the active profile may write it is decided by manifest.py
# alone, from the configuration in force — this script deliberately knows no
# profile name and reads only the exit code:
#
#     0  already current            -> report, write nothing
#     1  dirty or missing           -> build it, report the new root
#     3  refused by design          -> compute and report; the step is still OK
#     2  error                      -> the step FAILs
#
# The profile is propagated so manifest.py sees the same configuration the
# index step used; the refusal, if any, is manifest.py's to make.

manifest_py() {
    RAG_PROFILE="$PROFILE" RAG_LOG_QUIET=1 "$VENV_PY" "$RAG_ROOT/manifest.py" \
        "$@" --root "$KB_DIR" 2>&1
}

step_begin
DIFF_OUT="$(manifest_py --diff)"
DIFF_RC=$?
case "$DIFF_RC" in
    0)
        step_end "manifest" "OK" "already current, not rewritten - $(echo "$DIFF_OUT" | tail -1)"
        ;;
    1)
        BUILD_OUT="$(manifest_py)"
        BUILD_RC=$?
        if [ $BUILD_RC -ne 0 ]; then
            step_end "manifest" "FAIL" "manifest.py exited $BUILD_RC: $(echo "$BUILD_OUT" | tail -1)"
            skip_rest "mcp"
        fi
        step_end "manifest" "OK" "rebuilt - $(echo "$BUILD_OUT" | tail -1)"
        ;;
    3)
        COMPUTE_OUT="$(manifest_py --compute)"
        COMPUTE_RC=$?
        if [ $COMPUTE_RC -ne 0 ]; then
            step_end "manifest" "FAIL" "manifest.py --compute exited $COMPUTE_RC: $(echo "$COMPUTE_OUT" | tail -1)"
            skip_rest "mcp"
        fi
        # The refusal's first sentence is the reason; the rest is advice for a
        # human at a prompt and does not belong in a one-line health report.
        REASON="$(echo "$DIFF_OUT" | tail -1 | sed -e 's/^refusing: //' -e 's/\. .*$//')"
        ROOT_LINE="$(echo "$COMPUTE_OUT" | tail -1 | sed 's/ - computed, not written$//')"
        step_end "manifest" "OK" "computed, not written - $ROOT_LINE ($REASON)"
        ;;
    *)
        step_end "manifest" "FAIL" "manifest.py --diff exited $DIFF_RC: $(echo "$DIFF_OUT" | tail -1)"
        skip_rest "mcp"
        ;;
esac

fi  # end of the "a knowledge base is present" branch (steps 4-5)

# --- refresh: put the resulting root in the log -----------------------------
#
# Every refresh logs a root line, including one that changed nothing. The line
# answers "what is the root as of this refresh?", which is a different question
# from "did anything move?" — and the first is the one you ask after
# editing a note.

if [ "$REFRESH" = "1" ]; then
    RAG_PROFILE="$PROFILE" RAG_LOG_QUIET=1 "$VENV_PY" - "$RAG_ROOT" "$PROFILE" <<'PY' >/dev/null 2>&1
import sys
sys.path.insert(0, sys.argv[1])
import manifest
from raglog import get_logger

built, _resolver = manifest.build()
stored = manifest.load_stored() or {}
get_logger("refresh").info(
    "refresh: profile=%s root=%s files=%d chunks=%d (published root %s)",
    sys.argv[2],
    built["root"],
    built["file_count"],
    built["chunk_count"],
    stored.get("root", "none"),
)
PY
fi

# --- step 6: MCP handshake --------------------------------------------------

if [ "$REFRESH" = "1" ]; then
    step_skip "mcp" "refresh: handshake not run"
    report_and_exit
fi

step_begin
MCP_OUT="$RAG_ROOT/logs/.install-mcp-stdout.$$"
MCP_ERR="$RAG_ROOT/logs/.install-mcp-stderr.$$"
{
    printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"install.sh","version":"1"}}}'
    printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
} | RAG_LOG_QUIET=1 "$VENV_PY" "$RAG_ROOT/mcp_server.py" > "$MCP_OUT" 2> "$MCP_ERR"
MCP_RC=$?

MCP_VERDICT="$("$VENV_PY" - "$MCP_OUT" "$MCP_ERR" <<'PY' 2>&1
import json, sys

out_path, err_path = sys.argv[1], sys.argv[2]
lines = [l for l in open(out_path, encoding="utf-8").read().splitlines() if l.strip()]
stderr = open(err_path, encoding="utf-8").read()

problems = []
if len(lines) != 2:
    problems.append("expected 2 responses (the notification must stay silent), got %d" % len(lines))
messages = []
for line in lines:
    try:
        messages.append(json.loads(line))
    except ValueError as exc:
        problems.append("non-JSON on stdout: %s" % exc)

server = tools = protocol = None
if len(messages) == 2:
    init, listed = messages
    result = init.get("result") or {}
    info = result.get("serverInfo") or {}
    protocol = result.get("protocolVersion")
    server = "%s %s" % (info.get("name"), info.get("version"))
    if init.get("id") != 1:
        problems.append("initialize reply had id %r" % init.get("id"))
    if not protocol:
        problems.append("initialize reply carried no protocolVersion")
    tool_list = ((listed.get("result") or {}).get("tools")) or []
    tools = [t.get("name") for t in tool_list]
    if listed.get("id") != 2:
        problems.append("tools/list reply had id %r" % listed.get("id"))
    for required in ("search_toolbox", "reindex_toolbox"):
        if required not in tools:
            problems.append("tools/list is missing %s" % required)
if stderr.strip():
    problems.append("server wrote %d bytes to stderr" % len(stderr))

if problems:
    print("FAIL " + "; ".join(problems))
else:
    print("OK %s, protocol %s, tools: %s" % (server, protocol, ", ".join(tools)))
PY
)"
rm -f "$MCP_OUT" "$MCP_ERR"

if [ $MCP_RC -ne 0 ]; then
    step_end "mcp" "FAIL" "server exited $MCP_RC"
elif [ "${MCP_VERDICT#FAIL}" != "$MCP_VERDICT" ]; then
    step_end "mcp" "FAIL" "${MCP_VERDICT#FAIL }"
else
    step_end "mcp" "OK" "${MCP_VERDICT#OK }"
fi

report_and_exit
