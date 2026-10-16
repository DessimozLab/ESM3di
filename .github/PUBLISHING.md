# Python package releases

The [package workflow](workflows/publish.yml) builds `esm3di` and
`esm3di-training`, checks their metadata and disjoint wheel contents, and tests
installed wheels outside the checkout on Python 3.10, 3.11 and 3.12. It also
checks the inference-only installation before adding training dependencies.
CI tests are offline and use small test models; they do not download Hub weights
or replace the separate real-checkpoint compatibility checks.

Pull requests and pushes to `master` or `harmonize-v2-inference-learning` only
build and test. Publishing uses the same validated artifacts, in separate jobs
with OIDC permission and no source checkout. Actions are pinned to commit SHAs.

## Release hold: Python MAD licensing

**Publishing is currently disabled.** The provenance and redistribution license
of the bundled `src/esm3di/workflow/madroot/mad.py` still need confirmation from
Finn. The script identifies itself as MAD 2.2 (27 March 2018), references Giddy
Landan, and cites DOI:10.1038/s41559-017-0193.

The original C++ MADroot repository supplied by David is
<https://github.com/davidjamesbryant/MADroot>, licensed LGPL-3.0. That does not
establish the license of this different Python implementation. Resolve the
Python provenance, license and required notices before publishing to either
index. Leave the repository variable `PACKAGE_PUBLISHING_ENABLED` unset or
`false` until this is resolved. Do not add PyPI API tokens.

## One-time setup

1. On GitHub, create environments named `pypi` and `testpypi`. Configure required
   reviewer approval and restrict eligible release tags/branches as appropriate.
   Use a `v*` tag rule for `pypi`; permit the chosen release branch in `testpypi`.
2. On both PyPI and TestPyPI, configure a GitHub Trusted Publisher for **each**
   project (`esm3di` and `esm3di-training`):

   | Field | Value |
   | --- | --- |
   | Owner | `DessimozLab` |
   | Repository | `ESM3di` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` on PyPI; `testpypi` on TestPyPI |

   For projects that do not exist yet, add a pending publisher from your account's
   Publishing page. Pending publishers do not reserve project names.
   See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
3. After licensing is resolved and environments/publishers are configured, set
   the **repository-level** Actions variable `PACKAGE_PUBLISHING_ENABLED` to
   exactly `true`. This job-level guard must not be an environment-only variable.

The workflow does not create these external settings or enable publishing.

## Release procedure

1. Update both project versions together in the root and training
   `pyproject.toml` files, the exact inter-package dependency pins, and
   `src/esm3di/__init__.py`. Commit and push. Release versions are immutable on
   PyPI; use a new version for subsequent releases.
2. Run **Build, test and publish Python packages** manually from the intended
   release branch to upload to **TestPyPI**. With the publishing variable
   disabled, this only builds and tests. GitHub requires the workflow to be
   present on the default branch before offering manual dispatch.
3. Verify installation from TestPyPI in a disposable environment. Download only
   these two projects from TestPyPI, then resolve their dependencies from PyPI:

   ```bash
   python -m pip download --no-deps --only-binary=:all: \
     --index-url https://test.pypi.org/simple/ --dest test-wheels \
     esm3di==VERSION esm3di-training==VERSION
   python -m pip install test-wheels/*.whl
   python -m pip check
   esm3di --help
   esm3di-learn --help
   ```

   Replace `VERSION` with the matching published version. Check inference-only
   installation separately in a second environment.
4. Create the matching `vVERSION` tag and publish a GitHub release for that tag.
   The workflow verifies the tag against both package versions, reruns all
   checks, and requests approval through the `pypi` environment before upload.
   It publishes both distributions from that same run.

No tag, release, pull request, or ordinary push can upload while the publishing
variable is disabled. Uploads of two projects are not atomic: if an upload only
partly succeeds, inspect the index before retrying. The workflow deliberately
does not silently skip existing artifacts.
