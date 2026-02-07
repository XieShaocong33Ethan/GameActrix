from __future__ import annotations

import ast
import operator
from typing import Any, Callable


class CalculatorEvalError(ValueError):
    pass


_SAFE_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "min": min,
    "max": max,
}

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculator_eval(expression: str) -> float:
    """Safely evaluate a pure arithmetic expression.

    Allowed:
    - Numbers (int/float)
    - +, -, *, /, parentheses
    - abs(...), min(...), max(...)
    """
    expr = str(expression or "").strip()
    if not expr:
        raise CalculatorEvalError("empty expression")
    if len(expr) > 1024:
        raise CalculatorEvalError("expression too long")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover
        raise CalculatorEvalError("invalid expression") from exc

    value = _eval_ast(tree.body)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CalculatorEvalError("expression did not evaluate to a number")
    return float(value)


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorEvalError("only int/float constants are allowed")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        op = _BIN_OPS.get(op_type)
        if op is None:
            raise CalculatorEvalError(f"unsupported binary operator: {op_type.__name__}")
        return op(_eval_ast(node.left), _eval_ast(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        op = _UNARY_OPS.get(op_type)
        if op is None:
            raise CalculatorEvalError(f"unsupported unary operator: {op_type.__name__}")
        return op(_eval_ast(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorEvalError("only simple function calls are allowed")
        name = str(node.func.id)
        fn = _SAFE_FUNCS.get(name)
        if fn is None:
            raise CalculatorEvalError(f"unsupported function: {name}")
        if node.keywords:
            raise CalculatorEvalError("keyword arguments are not allowed")
        args = [_eval_ast(a) for a in node.args]
        return float(fn(*args))

    if isinstance(node, ast.Name):
        raise CalculatorEvalError(f"unknown name: {node.id}")

    raise CalculatorEvalError(f"unsupported syntax: {type(node).__name__}")
