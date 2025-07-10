from sympy import Symbol, Eq
from sympy.parsing.sympy_parser import parse_expr
import re

def extract_vars(equation_str):
    return re.findall(r'"(.*?)"', equation_str)

def replace_vars(equation_str, var_list):
    mapping = {}
    safe_expr = equation_str
    for i, var in enumerate(var_list):
        safe_name = f'v{i}'
        mapping[safe_name] = Symbol(var)
        safe_expr = safe_expr.replace(f'"{var}"', safe_name)
    return safe_expr, mapping

def parse_equations(equation_lines):
    parsed_equations = []
    dependencies = []

    for i, line in enumerate(equation_lines):
        # Skip comments and empty lines
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Extract quoted variable names
        quoted_vars = extract_vars(line)

        # Replace with safe variable names
        safe_expr, symbol_map = replace_vars(line, quoted_vars)

        try:
            lhs_str, rhs_str = safe_expr.split("=")
            lhs = parse_expr(lhs_str.strip(), local_dict=symbol_map)
            rhs = parse_expr(rhs_str.strip(), local_dict=symbol_map)
            equation = Eq(lhs, rhs)
            parsed_equations.append(equation)

            # Map back to original variable names
            lhs_name = str(lhs)
            rhs_deps = [str(s) for s in rhs.free_symbols]
            dependencies.append((lhs_name, rhs_deps))

        except Exception as e:
            print(f"Error parsing line: {line}")
            raise e

    return dependencies, parsed_equations