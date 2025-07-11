from sympy import Symbol, Eq, sympify, Piecewise, symbols, And
from sympy.parsing.sympy_parser import parse_expr
import re
import random

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

    for line in equation_lines:
        # Skip comments and empty lines
        if not line.strip() or line.strip().startswith("#"):
            continue

        if " iff " in line:
            result = parse_conditional_assignment(line)
            if result:
                lhs_sym, rhs_expr = result
                equation = Eq(lhs_sym, rhs_expr)
                parsed_equations.append(equation)
                dependencies.append((str(lhs_sym), [str(s) for s in rhs_expr.free_symbols]))
            else:
                print(f"Could not parse conditional line: {line}")
            continue  # Skip further processing for this line

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

def initialize_exogenous(equations, state={}, seed=0):
    if seed is not None:
        random.seed(seed)

    # 1. Collect all symbols used in RHS expressions
    all_rhs_symbols = set()
    lhs_symbols = set()

    for eq in equations:
        if isinstance(eq, Eq):
            lhs, rhs = eq.lhs, eq.rhs
        else:
            raise TypeError("Expecting sympy objects of type Eq!")
        all_rhs_symbols.update(rhs.free_symbols)
        lhs_symbols.add(lhs)

    # 2. Identify variable names that are exogenous
    known_names = set(state.keys())
    exogenous_symbols = [
        s for s in all_rhs_symbols
        if str(s) not in known_names and s not in lhs_symbols
    ]
    # print(f'The exogenous variables are:')
    # for ex_s in sorted([str(s) for s in exogenous_symbols]):
    #     print(f'"{ex_s}" = ')

    # 3. Initialize them with random values (e.g. between 0 and 1)
    new_state = state.copy()
    for sym in exogenous_symbols:
        new_state[str(sym)] = random.uniform(0.1, 0.9)

    return new_state

def parse_conditional_assignment(line):
    match = re.match(r'"([^"]+)"\s*=\s*(.+?)\s+iff\s+(.+)', line.strip())
    if not match:
        return None

    var_name, value_expr, condition_expr = match.groups()

    # Convert string names to sympy symbols
    symbols_in_expr = re.findall(r'"([^"]+)"', line)
    symbol_map = {name: symbols(name) for name in symbols_in_expr}

    # Replace "var" → symbol in expressions
    for name, sym in symbol_map.items():
        value_expr = value_expr.replace(f'"{name}"', name)
        condition_expr = condition_expr.replace(f'"{name}"', name)

    # Convert to sympy expressions
    value = sympify(value_expr, locals=symbol_map)
    condition = handle_condition(condition_expr, symbol_map)

    # Return sympy Piecewise assignment
    lhs_sym = symbol_map[var_name]
    rhs_expr = Piecewise((value, condition), (0, True))
    return lhs_sym, rhs_expr

def handle_condition(expr_str, symbol_map):
    # Detect chained comparison (e.g., "a <= b <= c")
    match = re.match(r'(.+?)\s*(<=|>=|<|>)\s*(.+?)\s*(<=|>=|<|>)\s*(.+)', expr_str)
    if match:
        a, op1, b, op2, c = match.groups()
        expr1 = parse_expr(f"{a.strip()} {op1} {b.strip()}", local_dict=symbol_map)
        expr2 = parse_expr(f"{b.strip()} {op2} {c.strip()}", local_dict=symbol_map)
        return And(expr1, expr2)
    else:
        return parse_expr(expr_str, local_dict=symbol_map)
