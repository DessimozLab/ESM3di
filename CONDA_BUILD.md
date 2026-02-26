# Building Conda Package for ESM3di

## Prerequisites

Install conda-build:
```bash
conda install conda-build anaconda-client
```

## Build the Package

### Option 1: Build Locally

From the project root directory:

```bash
conda build conda-recipe
```

This will create a package in your conda-bld directory (usually `~/miniconda3/conda-bld/` or similar).

### Option 2: Build for Multiple Platforms

```bash
# Build for current platform
conda build conda-recipe

# Convert to other platforms (no compilation needed for noarch)
conda convert --platform all ~/miniconda3/conda-bld/noarch/esm3di-0.1.0-py_0.tar.bz2 -o outputdir/
```

## Test the Package Locally

After building, install locally to test:

```bash
# Find the package path
conda build conda-recipe --output

# Install from local build
conda install --use-local esm3di

# Test the CLI tools
esm3di-train --help
esm3di-makefoldseekdb --help
esm3di-buildtrainingset --help
esm3di-download-alphafold --help
```

## Upload to Anaconda Cloud

1. Create an account at https://anaconda.org
2. Login via CLI:
   ```bash
   anaconda login
   ```

3. Upload your package:
   ```bash
   anaconda upload ~/miniconda3/conda-bld/noarch/esm3di-0.1.0-py_0.tar.bz2
   ```

4. Users can then install with:
   ```bash
   conda install -c yourusername esm3di
   ```

## Build for Conda-Forge (Optional)

For wider distribution via conda-forge:

1. Fork https://github.com/conda-forge/staged-recipes
2. Add your recipe to `recipes/esm3di/`
3. Submit a pull request
4. Once merged, your package will be available via:
   ```bash
   conda install -c conda-forge esm3di
   ```

## Development Workflow

For development, use editable install:
```bash
conda create -n esm3di-dev python=3.10
conda activate esm3di-dev
pip install -e .
```

## Updating the Package

1. Update version in `pyproject.toml` and `conda-recipe/meta.yaml`
2. Rebuild: `conda build conda-recipe`
3. Upload new version: `anaconda upload <path-to-new-package>`

## Notes

- The recipe uses `noarch: python` since this is pure Python code
- Dependencies are managed through conda channels
- PyTorch should be installed from conda-forge or pytorch channel
- Entry points are automatically created from the recipe
