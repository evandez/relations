from invoke import Collection, task

# Root invoke namespace.
ns = Collection()


###############
# Setup       #
###############


@task
def install(c):
    """Install the project into the current environment."""
    c.run("python -m pip install --upgrade pip")
    c.run("python -m pip install -r requirements.txt")


ns.add_task(install)

###############
# Code health #
###############


@task
def presubmit_black(c, fix=False):
    """Lint with black."""
    cmd = "python -m black src scripts --config pyproject.toml"
    if not fix:
        cmd += " --check"
    c.run(cmd)


@task
def presubmit_isort(c, fix=False):
    """Lint with isort."""
    cmd = "python -m isort src scripts"
    if not fix:
        cmd += " --check"
    c.run(cmd)


@task
def presubmit_mypy(c):
    """Run mypy type checker."""
    c.run("python -m mypy src scripts")


@task
def presubmit_pytest(c):
    """Run pytest for all unit tests."""
    c.run("python -m pytest tests")


@task
def presubmit(c, fix=False):
    """Run lint, testing, and type checking."""
    presubmit_black(c, fix=fix)
    presubmit_isort(c, fix=fix)
    presubmit_mypy(c)
    presubmit_pytest(c)


ns_presubmit = Collection("presubmit")
ns_presubmit.add_task(presubmit_black, "black")
ns_presubmit.add_task(presubmit_isort, "isort")
ns_presubmit.add_task(presubmit_mypy, "mypy")
ns_presubmit.add_task(presubmit_pytest, "pytest")
ns_presubmit.add_task(presubmit, default=True)

ns.add_collection(ns_presubmit)

###############
# Experiments #
###############
import experiments

ns_x = Collection.from_module(experiments, name="x")
ns.add_collection(ns_x)
