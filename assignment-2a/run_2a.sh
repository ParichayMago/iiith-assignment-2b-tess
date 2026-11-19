#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'Usage: bash run_2a.sh Assignment_2.zip NEW_OUTPUT_DIR [MODEL_DIR]' >&2
    exit 2
fi

project=$(cd -- "$(dirname -- "$0")" && pwd)
archive=$1
destination=$2
review_python=${REVIEW_PYTHON:-python3}
mfa=${MFA_EXECUTABLE:-$(command -v mfa)}
if [[ -e "$destination" ]]; then
    echo "Choose a new output directory: $destination" >&2
    exit 2
fi

# MFA 3.3.10 quotes paths with spaces before starting some Kaldi programs.
# Use a short staging path and a short alias for the conda bin directory.
stage=$(mktemp -d /tmp/iiith-2a.XXXXXX)
trap 'rm -rf -- "$stage"' EXIT
ln -s "$(cd -- "$(dirname -- "$mfa")" && pwd)" "$stage/bin"
export PATH="$stage/bin:$PATH"
export MFA_ROOT_DIR="$stage/mfa"
mkdir "$stage/models" "$stage/results"

releases=https://github.com/MontrealCorpusTools/mfa-models/releases/download
if [[ $# -eq 3 ]]; then
    cp "$3/english_mfa.zip" "$3/english_us_mfa.dict" "$stage/models/"
else
    curl -fLsS --retry 2 "$releases/acoustic-english_mfa-v3.1.0/english_mfa.zip" -o "$stage/models/english_mfa.zip"
    curl -fLsS --retry 2 "$releases/dictionary-english_us_mfa-v3.1.0/english_us_mfa.dict" -o "$stage/models/english_us_mfa.dict"
fi
(
    cd "$stage/models"
    shasum -a 256 -c <<'EOF'
2c08bd4f82c3943dd57ac09aeac99dbce17a1e1bfe9fd932c8d95cd13d971068  english_mfa.zip
cbf4ed8cee1338e009b9b387bcb9ee789ce70b98c61ca9bc6ed4f4e63d4812d5  english_us_mfa.dict
EOF
)

"$review_python" "$project/scripts/prepare_corpus.py" "$archive" "$stage/corpus"
# The only occurrence of "bed" is phrase-final. Keep its canonical stop variant.
awk '$1 != "bed" || $NF == "d"' "$stage/models/english_us_mfa.dict" > "$stage/models/assignment_2a.dict"
cat "$project/lexicon_additions.dict" >> "$stage/models/assignment_2a.dict"
"$stage/bin/mfa" find_oovs "$stage/corpus" "$stage/models/english_us_mfa.dict" "$stage/stock_oovs" --clean --num_jobs 2
"$stage/bin/mfa" find_oovs "$stage/corpus" "$stage/models/assignment_2a.dict" "$stage/final_oovs" --clean --num_jobs 2
cp "$stage/stock_oovs/oovs_found_english_us_mfa.txt" "$stage/results/stock_oovs.txt"
if [[ -s "$stage/final_oovs/oovs_found_assignment_2a.txt" ]]; then
    echo 'Unresolved vocabulary remains after applying the dictionary additions.' >&2
    cat "$stage/final_oovs/oovs_found_assignment_2a.txt" >&2
    exit 1
fi

"$stage/bin/mfa" align "$stage/corpus" "$stage/models/assignment_2a.dict" "$stage/models/english_mfa.zip" \
    "$stage/results/baseline" --clean --num_jobs 2
"$review_python" "$project/scripts/prepare_corpus.py" "$archive" "$stage/segmented-corpus" --segment-speech
"$stage/bin/mfa" align "$stage/segmented-corpus" "$stage/models/assignment_2a.dict" "$stage/models/english_mfa.zip" \
    "$stage/results/textgrids" --clean --num_jobs 2
"$review_python" "$project/scripts/inspect_alignments.py" "$stage/corpus" "$stage/results/textgrids" "$stage/results" \
    --baseline "$stage/results/baseline"
mkdir -p -- "$(dirname -- "$destination")"
mkdir -- "$destination"
cp -R "$stage/results/." "$destination/"
echo "Wrote Assignment 2a results to $destination"
