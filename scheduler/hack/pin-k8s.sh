#!/usr/bin/env bash
# k8s.io/kubernetes is not meant to be imported as a library: its go.mod points every
# k8s.io/* staging module at a local path. Out-of-tree schedulers must pin each staging
# module to the matching v0.X.Y release. This rewrites go.mod to do that.
#
#   hack/pin-k8s.sh 1.31.4
set -euo pipefail

VERSION=${1:?usage: $0 <kubernetes version, e.g. 1.31.4>}
VERSION=${VERSION#v}
STAGING="v0.${VERSION#1.}"

cd "$(dirname "$0")/.."
MODS=$(curl -fsSL "https://raw.githubusercontent.com/kubernetes/kubernetes/v${VERSION}/go.mod" \
  | sed -n 's|^[[:space:]]*\(k8s.io/[^[:space:]]*\) => ./staging/src/.*|\1|p')

for MOD in $MODS; do
  go mod edit "-replace=${MOD}=${MOD}@${STAGING}"
done
go get "k8s.io/kubernetes@v${VERSION}"
go mod tidy
echo "pinned k8s.io/kubernetes v${VERSION} (staging ${STAGING})"
