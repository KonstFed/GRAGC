#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash ragc/datatools/flatten_docstring_cache.bash repobench_docstring_graphs.zip
# or, if you already have the extracted folder:
#   DOCSTRING_CACHE_DIR=data/repobench/docstring_cache bash ragc/datatools/flatten_docstring_cache.bash

ZIP_PATH="${1:-}"

SRC_ROOT="${DOCSTRING_CACHE_DIR:-data/repobench/docstring_cache}"
DST_ROOT="data/repobench/parsed_graphs"

if [[ -n "${ZIP_PATH}" ]]; then
  echo "Extracting '${ZIP_PATH}' into '${SRC_ROOT}'..."
  mkdir -p "${SRC_ROOT}"
  unzip -q -o "${ZIP_PATH}" -d "${SRC_ROOT}"
fi

mkdir -p "${DST_ROOT}"

echo "Flattening GML graphs from '${SRC_ROOT}' into '${DST_ROOT}'..."

for repo_dir in "${SRC_ROOT}"/*; do
  [ -d "${repo_dir}" ] || continue

  repo_name="$(basename "${repo_dir}")"

  gml_file="$(find "${repo_dir}" -maxdepth 1 -type f -name '*.gml' | head -n 1 || true)"

  if [ -z "${gml_file}" ]; then
    echo "Warning: no .gml file found in '${repo_dir}', skipping" >&2
    continue
  fi

  dst_path="${DST_ROOT}/${repo_name}.gml"

  echo "Copying '${gml_file}' -> '${dst_path}'"
  cp "${gml_file}" "${dst_path}"
done

echo "Done. You can now point 'graphs_path' to '${DST_ROOT}'."

