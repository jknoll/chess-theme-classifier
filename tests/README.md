# Chess Theme Classifier Tests

This directory contains tests for the Chess Theme Classifier project.

## Running Tests

To run the basic tests (which don't incur ISC costs):

```bash
cd /root/chess-theme-classifier
python -m pytest tests/
```

## ISC Tests

Some tests are marked as "ISC" tests because they interact with the Strong Compute infrastructure and may incur costs. These tests are skipped by default.

To run ISC tests, you need to explicitly enable them:

```bash
cd /root/chess-theme-classifier
python -m pytest tests/ --run-isc
```

⚠️ **Warning**: Running ISC tests may incur costs as they interact with training clusters. Only run these when necessary and with proper authorization.

## Available Test Files

- `test_train.py`: Tests for the basic train.py script functionality
- `test_train_isc.py`: Tests for ISC-specific functionality (skipped by default)

## Adding New Tests

When adding new tests:

1. If your tests might incur costs on ISC infrastructure, mark them with `@pytest.mark.isc` or place them in files that use `pytestmark = pytest.mark.isc`
2. Never include tests that automatically run `isc train` commands, as these will definitely incur costs
3. For expensive operations, use pytest fixtures to avoid repeating setup code

## Test Configuration

The test configuration is in `conftest.py` which includes:

- Command line options for controlling which tests run
- Fixtures for common test resources
- Test collection customization