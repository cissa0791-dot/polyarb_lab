from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


Assignment = Mapping[str, bool]


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class Constraint:
    name: str

    def evaluate(self, assignment: Assignment) -> ConstraintResult:
        raise NotImplementedError


@dataclass(frozen=True)
class ImplicationConstraint(Constraint):
    premise: str
    consequence: str

    def evaluate(self, assignment: Assignment) -> ConstraintResult:
        premise_value = bool(assignment.get(self.premise, False))
        consequence_value = bool(assignment.get(self.consequence, False))
        passed = (not premise_value) or consequence_value
        return ConstraintResult(self.name, passed, f"{self.premise}->{self.consequence}")


@dataclass(frozen=True)
class MutualExclusionConstraint(Constraint):
    symbols: Sequence[str]

    def evaluate(self, assignment: Assignment) -> ConstraintResult:
        true_count = sum(1 for symbol in self.symbols if assignment.get(symbol, False))
        passed = true_count <= 1
        return ConstraintResult(self.name, passed, f"true_count={true_count}")


@dataclass(frozen=True)
class ExactlyOneConstraint(Constraint):
    symbols: Sequence[str]

    def evaluate(self, assignment: Assignment) -> ConstraintResult:
        true_count = sum(1 for symbol in self.symbols if assignment.get(symbol, False))
        passed = true_count == 1
        return ConstraintResult(self.name, passed, f"true_count={true_count}")


@dataclass(frozen=True)
class SubsetConstraint(Constraint):
    subset_symbol: str
    superset_symbol: str

    def evaluate(self, assignment: Assignment) -> ConstraintResult:
        subset_value = bool(assignment.get(self.subset_symbol, False))
        superset_value = bool(assignment.get(self.superset_symbol, False))
        passed = (not subset_value) or superset_value
        return ConstraintResult(self.name, passed, f"{self.subset_symbol}<= {self.superset_symbol}")


@dataclass
class ConstraintGraph:
    constraints: list[Constraint] = field(default_factory=list)

    def add(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def evaluate(self, assignment: Assignment) -> list[ConstraintResult]:
        return [constraint.evaluate(assignment) for constraint in self.constraints]


@dataclass(frozen=True)
class FeasibilityCheck:
    passed: bool
    violations: list[ConstraintResult]


class FeasibilityChecker:
    def check(self, graph: ConstraintGraph, assignment: Assignment) -> FeasibilityCheck:
        results = graph.evaluate(assignment)
        violations = [result for result in results if not result.passed]
        return FeasibilityCheck(passed=not violations, violations=violations)

