Contributions are welcome!

If you have a new feature/bug report, make sure you create an [issue](https://github.com/htjb/margarine/issues), and consult existing ones first, in case your suggestion is already being addressed.

If you want to go ahead and create the feature yourself, you should fork the repository to you own github account and create a new branch with an appropriate name. Commit any code modifications to that branch, push to GitHub, and then create a pull request via your forked repository.

## Contributing - `pre-commit`

To try and maintain a consistent style the code base is using pre-commit, ruff and isort. If you are ready to contribute to `margarine` please follow the instructions below before making a PR.

First, ensure that pre-commit is installed:
```
pip install pre-commit
```
Then install the pre-commit to the .git folder:
```
pre-commit install
```
Before running `git commit` you should run `pre-commit run --files your_file` on any additional or changed files. This will check the code against ruff and isort and make any appropriate changes.