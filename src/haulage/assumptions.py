from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Assumption:
    r"""An identifying assumption with a formal statement and (optional) testable implication.

    Attributes:
        name: Short tag, e.g. "parallel_trends".
        latex: Formal LaTeX statement.
        description: Plain-English description.
        testable: Whether the assumption has any empirically testable implication.
        test: Callable that consumes data and returns a diagnostic dict (p-value, stat, etc.).
        references: Canonical citations.
    """

    name: str
    latex: str
    description: str
    testable: bool = False
    test: Callable[..., dict[str, Any]] | None = None
    references: tuple[str, ...] = field(default_factory=tuple)

    def run_test(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Invoke the diagnostic, or raise if the assumption is untestable."""
        if self.test is None:
            raise ValueError(f"assumption {self.name!r} has no testable implementation")
        return self.test(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class AssumptionSet:
    """A bundle of assumptions attached to a particular estimator.

    Attributes:
        assumptions: The individual assumptions.
        estimator: Short tag for the estimator these attach to.
    """

    assumptions: tuple[Assumption, ...]
    estimator: str

    def __iter__(self) -> Any:
        return iter(self.assumptions)

    def __len__(self) -> int:
        return len(self.assumptions)

    def get(self, name: str) -> Assumption:
        """Look up an assumption by name."""
        for a in self.assumptions:
            if a.name == name:
                return a
        raise KeyError(name)

    @property
    def testable(self) -> tuple[Assumption, ...]:
        return tuple(a for a in self.assumptions if a.testable)

    @property
    def untestable(self) -> tuple[Assumption, ...]:
        return tuple(a for a in self.assumptions if not a.testable)

    def to_markdown(self) -> str:
        """Render the set as a GitHub-flavored markdown table."""
        lines = ["| Assumption | Testable | Statement |", "|---|---|---|"]
        for a in self.assumptions:
            stmt = a.latex.replace("|", r"\|")
            lines.append(f"| {a.name} | {'yes' if a.testable else 'no'} | ${stmt}$ |")
        return "\n".join(lines)


PARALLEL_TRENDS = Assumption(
    name="parallel_trends",
    latex=(
        r"\mathbb{E}[Y_t(\infty) - Y_{t-1}(\infty) \mid G=g, X] "
        r"= \mathbb{E}[Y_t(\infty) - Y_{t-1}(\infty) \mid G=\infty, X]"
    ),
    description="Conditional parallel counterfactual trends across cohorts.",
    testable=True,
    references=("Callaway & Sant'Anna (2021)", "Roth, Sant'Anna, Bilinski, Poe (2023)"),
)

NO_ANTICIPATION = Assumption(
    name="no_anticipation",
    latex=r"\mathbb{E}[Y_t(g) - Y_t(\infty) \mid G=g] = 0 \text{ for } t < g",
    description="Treatment has no causal effect prior to the treatment date.",
    testable=True,
    references=("Callaway & Sant'Anna (2021)",),
)

SUTVA = Assumption(
    name="SUTVA",
    latex=r"Y_i(D_i) \perp\!\!\!\perp D_{-i}",
    description="Stable unit treatment value: one unit's treatment does not affect others.",
    testable=True,
    references=("Rubin (1980)", "Cox (1958)"),
)

OVERLAP = Assumption(
    name="overlap",
    latex=r"0 < \mathbb{P}(D=1 \mid X=x) < 1",
    description="Both treatment arms have positive propensity at each covariate value.",
    testable=True,
    references=("Rosenbaum & Rubin (1983)",),
)

EXCLUSION = Assumption(
    name="IV_exclusion",
    latex=r"Y(d, z) = Y(d) \quad \forall\ z",
    description="The instrument affects the outcome only through the treatment.",
    testable=False,
    references=("Angrist, Imbens & Rubin (1996)",),
)

MONOTONICITY = Assumption(
    name="IV_monotonicity",
    latex=r"D_i(1) \ge D_i(0) \text{ for all } i",
    description="No defiers: the instrument never weakly discourages treatment.",
    testable=False,
    references=("Imbens & Angrist (1994)",),
)

RELEVANCE = Assumption(
    name="IV_relevance",
    latex=r"\mathrm{Cov}(Z, D) \neq 0",
    description="The instrument shifts the treatment; weak-IV diagnostics quantify strength.",
    testable=True,
    references=("Stock, Wright & Yogo (2002)", "Montiel Olea & Pflueger (2013)"),
)

NO_MANIPULATION = Assumption(
    name="no_manipulation",
    latex=r"f_X \text{ continuous at cutoff } c",
    description="Running variable density continuous at the cutoff (no precise sorting).",
    testable=True,
    references=("McCrary (2008)", "Cattaneo, Jansson & Ma (2020)"),
)

STABLE_SC_WEIGHTS = Assumption(
    name="SC_stability",
    latex=r"Y_{it}(0) = \sum_{j \in \mathcal{D}} w_j^* Y_{jt}(0)",
    description="Pre-period convex combination of donor units generalizes out of sample.",
    testable=True,
    references=("Abadie, Diamond & Hainmueller (2010)",),
)
