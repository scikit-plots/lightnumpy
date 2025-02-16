#!/bin/bash

# git config --global --unset-all safe.directory
# git config --global --get-all safe.directory

## Directories to mark as safe
for DIR in \
  "$(realpath ./)" \
  "$(realpath ./third_party/array-api-compat)" \
  "$(realpath ./third_party/array-api-extra)" \
  "$(realpath ./third_party/astropy)" \
  "$(realpath ./third_party/seaborn)"
do
  ## Try adding each directory
  git config --global --add safe.directory "$DIR" 2>/dev/null || FALLBACK=1
done

## If any command failed, allow all directories as safe
if [ "$FALLBACK" = "1" ]; then
  echo "Some directories failed. Allowing all directories as safe..."
  ## Alternative: Bypass Ownership Checks (If Safe)
  git config --global --add safe.directory '*'
fi

echo "Safe directory configuration complete."

# to initialise local config file and fetch + checkout submodule (not needed every time)
git submodule update --init --recursive  # download submodules

git remote add upstream https://github.com/scikit-plots/scikit-plots.git || true

git fetch upstream --tags

# Install the development version of scikit-plots
pip install -r ./requirements/build.txt
pip install --no-build-isolation --no-cache-dir -e .[dev,build,test,docs,gpu] -v

pip install pre-commit

( cd /workspaces/scikit-plots/ && pre-commit install)

echo "Continue to below section: Creating a branch"

echo "https://scikit-plots.github.io/dev/devel/quickstart_contributing.html#creating-a-branch"
