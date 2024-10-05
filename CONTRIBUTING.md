# Contributing to StockSense

First off, thank you for considering contributing to StockSense! It's people like you that make StockSense such a great tool. We welcome contributions from everyone, regardless of their level of experience.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
   - [Issues](#issues)
   - [Pull Requests](#pull-requests)
3. [Coding Standards](#coding-standards)
4. [Commit Messages](#commit-messages)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by the [StockSense Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@stocksense.ai](mailto:conduct@stocksense.ai).

## Getting Started

Contributions to StockSense are made via Issues and Pull Requests (PRs). A few general guidelines that cover both:

- Search for existing Issues and PRs before creating your own.
- We work hard to makes sure issues are handled in a timely manner but, depending on the impact, it could take a while to investigate the root cause. A friendly ping in the comment thread to the submitter or a contributor can help draw attention if your issue is blocking.

### Issues

Issues should be used to report problems with the library, request a new feature, or to discuss potential changes before a PR is created. When you create a new Issue, a template will be loaded that will guide you through collecting and providing the information we need to investigate.

If you find an Issue that addresses the problem you're having, please add your own reproduction information to the existing issue rather than creating a new one. Adding a [reaction](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/) can also help be indicating to our maintainers that a particular problem is affecting more than just the reporter.

### Pull Requests

PRs to our libraries are always welcome and can be a quick way to get your fix or improvement slated for the next release. In general, PRs should:

1. Only fix/add the functionality in question OR address wide-spread whitespace/style issues, not both.
2. Add unit or integration tests for fixed or changed functionality (if a test suite already exists).
3. Address a single concern in the least number of changed lines as possible.
4. Include documentation in the repo or on our [docs site](https://docs.stocksense.ai).
5. Be accompanied by a complete Pull Request template (loaded automatically when a PR is created).

For changes that address core functionality or would require breaking changes (e.g. a major release), it's best to open an Issue to discuss your proposal first. This is not required but can save time creating and reviewing changes.

In general, we follow the ["fork-and-pull" Git workflow](https://github.com/susam/gitpr)

1. Fork the repository to your own Github account
2. Clone the project to your machine
3. Create a branch locally with a succinct but descriptive name
4. Commit changes to the branch
5. Following any formatting and testing guidelines specific to this repo
6. Push changes to your fork
7. Open a PR in our repository and follow the PR template so that we can efficiently review the changes.

## Coding Standards

To ensure consistency throughout the source code, keep these rules in mind as you are working:

1. All features or bug fixes **must be tested** by one or more specs (unit-tests).
2. We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. Use `flake8` to check your Python code style.
3. We use [Google style](https://google.github.io/styleguide/pyguide.html) for documenting Python code.
4. We use [Black](https://github.com/psf/black) for code formatting. Run `black .` before committing.

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This leads to **more readable messages** that are easy to follow when looking through the **project history**.

## Testing

We use pytest for our test suite. To run the tests, use:

```
pytest
```

Make sure to write tests for new features and bug fixes. Our CI pipeline will run these tests for all PRs.

## Documentation

Documentation is a crucial part of this project. Please consider updating the documentation when you submit your PR.

- If you're adding a new feature, include documentation on how to use it.
- If you're fixing a bug, update any relevant documentation that could have been affected.

## Community

Discussions about StockSense take place on this repository's [Issues](https://github.com/yourusername/stocksense/issues) and [Pull Requests](https://github.com/yourusername/stocksense/pulls) sections. Anybody is welcome to join these conversations.

Wherever possible, do not take these conversations to private channels, including contacting the maintainers directly. Keeping communication public means everybody can benefit and learn from the conversation.
