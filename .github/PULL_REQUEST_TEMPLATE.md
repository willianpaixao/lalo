## Description

<!-- Provide a clear and concise description of what this PR does -->

## Motivation

<!-- Why is this change needed? What problem does it solve? -->

Fixes #(issue number)

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🎨 Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test update
- [ ] 🔧 Configuration/Build update

## Changes Made

<!-- List the specific changes made in this PR -->

- Change 1
- Change 2
- Change 3

## Testing

<!-- Describe the tests you ran and how to reproduce them -->

### Test Configuration

- **OS**: 
- **Python version**: 
- **GPU**: 

### Test Cases

<!-- Describe what testing you performed -->

```bash
# Example commands used for testing
pytest tests/test_new_feature.py -v
lalo convert test.epub --new-option
```

**Test results:**
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed


## Checklist

<!-- Mark completed items with an "x" -->

### Code Quality

- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have run `ruff check --fix` and `ruff format`
- [ ] I have run `mypy lalo/` with no errors
- [ ] My commits follow the [conventional commits](https://www.conventionalcommits.org/) format

### Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes (`pytest tests/ -v`)
- [ ] I have added integration tests if applicable

### Documentation

- [ ] I have updated the documentation to reflect my changes
- [ ] I have updated the README.md if needed
- [ ] I have added docstrings to new functions/classes

### Dependencies

- [ ] I have checked that no new dependencies are needed
- [ ] OR: I have added necessary dependencies to `pyproject.toml` with justification below

## Breaking Changes

<!-- If this introduces breaking changes, describe them here and provide migration instructions -->

**None** / **Describe breaking changes and migration path**

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Related Issues/PRs

<!-- Link to related issues or PRs -->

- Related to #
- Depends on #
- Blocks #
