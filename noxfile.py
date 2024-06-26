"""This module implements our CI function calls."""
import nox


@nox.session(name="test")
def run_test(session):
    """Run pytest."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest")


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "-m", "not slow")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("-r", "requirements.txt")
    session.install("pylint")
    session.run("pylint", "src", "script")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install("-r", "requirements.txt")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--strict",
        "--no-warn-return-any",
        "--explicit-package-bases",
        "--namespace-packages",
        "--implicit-reexport",  # tensorboard is untyped
        "--allow-untyped-calls",  # tensorboard is untyped
        "src/baseline_detection",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "script", "noxfile.py")
    session.run("black", "src", "script", "noxfile.py")
