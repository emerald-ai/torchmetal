# Contributing

You are here to help on torchmetal? Awesome, feel welcome and read the
following sections in order to know how to ask questions and how to work on
something.

Following these guidelines helps to communicate that you respect the time of
the developers managing and developing this open source project. In return,
they should reciprocate that respect in addressing your issue, assessing
changes, and helping you finalize your pull requests. All members of our
community are expected to be welcoming and friendly in all of our spaces.


## Get in touch

- Ask usage questions (“How do I?”) on [the repository discussions][discus].
- Report bugs, suggest features or view the source code [on GitHub][issues].
- Discuss topics on [the repository discussions][discus].

There are many ways to contribute, from writing tutorials or blog posts,
improving the documentation, submitting bug reports and feature requests or
writing code which can be incorporated into torchmetal itself. 

Please, don't use the issue tracker for support or machine learning questions.
Check whether [the repository questions and discussions][discus] can help with
your issue. Stack Overflow is also worth considering.


## Ground Rules

- Be welcoming to newcomers and encourage new contributors.
- Discuss things transparently and get community feedback.
  - Discuss changes and enhancements before starting work on them.
  - Create issues for any major changes and enhancements that you wish to make.
- Keep things as simple as possible.
  - Don't add any classes to the codebase unless absolutely needed.
  - Err on the side of using functions.
- Ensure all code is formatted with [black][black] and [isort][isort].


## How to report a bug

Open an issue using the bug issue template.


## How to suggest a feature or enhancement

Open an issue using the enhancement issue template.


## Things to know

Running tests that use datasets require setting environment variables as
follows.

```bash
export TORCHMETAL_DATA_FOLDER="$HOME/my/datasets/"
export TORCHMETAL_DOWNLOAD=true
```


[discus]: https://github.com/sevro/torchmetal/discussions
[issues]: https://github.com/sevro/torchmetal/issues
[black]: https://github.com/psf/black
[isort]: https://pycqa.github.io/isort/
