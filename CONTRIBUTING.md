# Contributing to Lalo

Thank you for your interest in contributing to Lalo! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
1. **Check existing issues** to avoid duplicates
2. **Update to the latest version** to see if the bug still exists
3. **Gather information** about your environment (OS, Python version, CUDA version, GPU model)

Use the **Bug Report** issue template and include:
- Clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- System information and error messages
- Sample EPUB file (if applicable and copyright-free)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Use the **Feature Request** template and include:
- Clear description of the proposed feature
- Use cases and benefits
- Possible implementation approaches
- Any relevant examples from other projects

### Contributing Code

We welcome code contributions! Areas where you can help:
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements
- Code refactoring

## Development Setup

### Prerequisites

- **Python 3.12+** installed
- **NVIDIA GPU** with CUDA support (for testing TTS functionality)
- **ffmpeg** for audio processing

## Pull Request Process

### Before Submitting

Ensure your PR:
- [ ] Follows the coding standards
- [ ] Includes tests for new functionality
- [ ] Passes all CI checks (lint, type-check, tests)
- [ ] Updates documentation if needed
- [ ] Has a clear, descriptive title
- [ ] References related issues (e.g., "Fixes #123")

### PR Title Format

Use conventional commit format:
- `feat: add M4B export with chapter markers`
- `fix: resolve EPUB parsing error with special characters`
- `docs: update installation instructions for macOS`
- `perf: optimize GPU batching for faster conversion`

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Specific change 1
- Specific change 2
- Specific change 3

## Testing
How was this tested? Include steps to reproduce.

## Screenshots (if applicable)
Add screenshots for UI/output changes

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG updated (for features/fixes)
- [ ] All CI checks passing
```

## Issue Guidelines

### Issue Labels

We use labels to categorize issues:

**Type:**
- `bug` - Something isn't working
- `feature` - New feature request
- `enhancement` - Improvement to existing feature
- `documentation` - Documentation improvements
- `question` - Questions about usage or functionality

**Priority:**
- `priority: critical` - Security issues, data loss, crashes
- `priority: high` - Major bugs, important features
- `priority: medium` - Minor bugs, nice-to-have features
- `priority: low` - Cosmetic issues, future enhancements

**Status:**
- `status: needs-triage` - Needs review and labeling
- `status: ready` - Ready for development
- `status: in-progress` - Currently being worked on
- `status: blocked` - Blocked by other issues/PRs
- `status: needs-info` - Waiting for more information

**Component:**
- `component: cli` - Command-line interface
- `component: tts` - Text-to-speech engine
- `component: epub` - EPUB parsing
- `component: audio` - Audio processing/export
- `component: testing` - Test suite
- `component: ci-cd` - CI/CD pipeline

**Special:**
- `breaking-change` - Breaks backward compatibility
- `dependencies` - Dependency updates
- `performance` - Performance improvements
- `security` - Security-related issues

Thank you for contributing to Lalo! 🎉
