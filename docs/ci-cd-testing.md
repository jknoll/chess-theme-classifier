# CI/CD Test Automation

This document describes how automated testing is configured for the chess-theme-classifier project.

## GitHub Actions Workflow

Tests are automatically run via GitHub Actions on every push to `main` and on every pull request targeting `main`.

### Workflow Configuration

The workflow is defined in `.github/workflows/test.yml`:

```yaml
name: Run Pytest

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest

    - name: Run tests
      run: |
        pytest tests/
```

### Trigger Events

The workflow runs on two events:

1. **Push to main**: Any commit pushed directly to the `main` branch triggers the test suite
2. **Pull Request to main**: Any PR targeting `main` triggers tests, allowing reviewers to verify changes before merging

### Workflow Steps

1. **Checkout**: Clones the repository using `actions/checkout@v4`
2. **Python Setup**: Installs Python 3.10 using `actions/setup-python@v5`
3. **Dependencies**: Installs project dependencies from `requirements.txt` plus pytest
4. **Test Execution**: Runs `pytest tests/` to execute all tests in the `tests/` directory

## Test Suite

The test suite is located in `tests/` and includes:

| Test File | Description |
|-----------|-------------|
| `test_train.py` | Tests for training script, argument parsing, model creation, dataset loading |
| `test_dataset_conditional_cache.py` | Tests for class-conditional caching functionality |

### Key Tests

- **test_parse_args**: Verifies command-line argument parsing
- **test_train_script_runs**: Runs train.py in test mode with single GPU to verify the training loop works
- **test_dataset_loads**: Verifies dataset loading from CSV files
- **test_model_creation**: Verifies model instantiation and forward pass
- **test_conditional_cache_includes_labels**: Verifies class-conditional augmentation caching

## Running Tests Locally

```bash
# Activate virtual environment
source .chess-theme-classifier/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_train.py -v

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=. --cov-report=html
```

## CI Badge

The README displays a CI status badge:

```markdown
![CI](https://github.com/jknoll/chess-theme-classifier/actions/workflows/test.yml/badge.svg)
```

This badge shows:
- **Green (passing)**: All tests passed on the latest commit
- **Red (failing)**: One or more tests failed
- **Yellow (pending)**: Tests are currently running

## Adding New Tests

1. Create test files in the `tests/` directory with the `test_` prefix
2. Follow pytest conventions for test functions (prefix with `test_`)
3. Tests run automatically on next push/PR

## Troubleshooting CI Failures

1. Check the GitHub Actions tab for detailed logs
2. Reproduce locally with the same Python version (3.10)
3. Ensure all dependencies are in `requirements.txt`
4. For GPU-dependent tests, use `--single_gpu` mode with mock environment variables
