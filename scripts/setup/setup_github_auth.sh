#!/usr/bin/env bash
set +x
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

origin_url=https://github.com/stoneboat/model_provenance_code.git
if [[ $(git remote get-url --push --all origin) != "$origin_url" ]]; then
    printf 'Expected origin to push to %s.\n' "$origin_url" >&2
    exit 1
fi

github_username=${GITHUB_USERNAME:-}
github_token=${GH_TOKEN:-${GITHUB_TOKEN:-}}
if [[ -z $github_username ]]; then
    read -r -p 'GitHub username: ' github_username
fi
if [[ -z $github_token ]]; then
    read -rs -p 'GitHub personal access token: ' github_token
    printf '\n' >&2
fi
if [[ ! $github_username =~ ^[A-Za-z0-9-]+$ || -z $github_token || $github_token =~ [[:space:][:cntrl:]] ]]; then
    printf 'Enter a valid GitHub username and personal access token.\n' >&2
    exit 1
fi

# Validate the credentials before replacing anything in the cache.
input_helper='!credentials() {
    if [ "$1" = get ]; then
        printf "username=%s\npassword=%s\n" "$MODEL_PROVENANCE_GIT_USERNAME" "$MODEL_PROVENANCE_GIT_TOKEN"
    fi
}; credentials'
MODEL_PROVENANCE_GIT_USERNAME="$github_username" \
MODEL_PROVENANCE_GIT_TOKEN="$github_token" \
GIT_TERMINAL_PROMPT=0 GIT_ASKPASS= \
    git -c credential.helper= -c "credential.helper=$input_helper" \
        push --dry-run --no-verify origin HEAD

cache_helper='cache --timeout=86400'
printf 'url=%s\nusername=%s\npassword=%s\n\n' "$origin_url" "$github_username" "$github_token" |
    git -c credential.helper= -c "credential.helper=$cache_helper" \
        -c credential.useHttpPath=true credential approve

git config --local "credential.$origin_url.useHttpPath" true
git config --local --replace-all "credential.$origin_url.helper" "$cache_helper"
printf 'url=%s\n\n' "$origin_url" |
    GIT_TERMINAL_PROMPT=0 GIT_ASKPASS= git credential fill >/dev/null
printf 'GitHub credentials cached for 24 hours or until the cache stops.\n'
