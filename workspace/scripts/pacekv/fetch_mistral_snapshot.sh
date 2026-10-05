#!/usr/bin/env bash
set -euo pipefail

# Fetch only the files required by the frozen 5090 R1a pilot. The consolidated
# checkpoint duplicates the sharded checkpoint and is intentionally omitted.
revision=c170c708c41dac9275d15a8fff4eca08d52bab71
destination=${1:?usage: fetch_mistral_snapshot.sh DESTINATION}
if [[ $(basename -- "$destination") != "$revision" ]]; then
  echo "destination must end in the exact model revision" >&2
  exit 2
fi
mkdir -p -- "$destination"
base="https://hf-mirror.com/mistralai/Mistral-7B-Instruct-v0.3/resolve/$revision"
metadata=(
  config.json
  generation_config.json
  model.safetensors.index.json
  special_tokens_map.json
  tokenizer.json
  tokenizer.model
  tokenizer.model.v3
  tokenizer_config.json
)
for file in "${metadata[@]}"; do
  curl --fail --location --silent --show-error --retry 4 --retry-delay 3 \
    --connect-timeout 20 --output "$destination/$file.part" \
    "$base/$file"
  mv -- "$destination/$file.part" "$destination/$file"
done

shards=(
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
)
hashes=(
  ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19
  8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0
  905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca
)
download_shard() {
  local file=$1 expected=$2
  if [[ -f "$destination/$file" ]] &&
      printf '%s  %s\n' "$expected" "$destination/$file" | sha256sum -c --status; then
    return 0
  fi
  curl --fail --location --silent --show-error --retry 4 --retry-delay 3 \
    --connect-timeout 20 --continue-at - --output "$destination/$file" \
    "$base/$file"
  printf '%s  %s\n' "$expected" "$destination/$file" | sha256sum -c --status
}
pids=()
for i in 0 1 2; do
  download_shard "${shards[$i]}" "${hashes[$i]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" != 0 ]]; then
  echo 'at least one model shard failed transfer or SHA256 verification' >&2
  exit 1
fi
for file in "${metadata[@]}" "${shards[@]}"; do
  test -s "$destination/$file"
done
(cd "$destination" && sha256sum "${metadata[@]}" "${shards[@]}")
