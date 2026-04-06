from sympy import Symbol, Eq, sympify, Piecewise, symbols, And
from sympy.parsing.sympy_parser import parse_expr
import re
import random
from equation_parser import build_matrix, check_matrix_consistency
import numpy as np

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

def print_equations(equations):
    print("\nThe equations are:")
    for i, eq in enumerate(equations):
        print(f'{i+1} -> {eq}')

def print_variabels(equations):
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

    # 2. Identify variable names that are exogenous/endogenous
    endogenous_symbols = [s for s in lhs_symbols]
    exogenous_symbols = [
        s for s in all_rhs_symbols
        if s not in lhs_symbols
    ]
    print(f'\nThe exogenous variables are:')
    for i, ex_s in enumerate(sorted([str(s) for s in exogenous_symbols])):
        print(f'{i+1} -> "{ex_s}"')
    # print(exogenous_symbols)
    print(f'\nThe endogenous variables are:')
    for i, end_s in enumerate(sorted([str(s) for s in endogenous_symbols])):
        print(f'{i+1} -> "{end_s}"')
    # print(endogenous_symbols)

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

# def initialize_variables(equations, state, seed=0):
#     random.seed(seed)

#     # 1. Collect all symbols
#     all_symbols = set()

#     for eq in equations:
#         if isinstance(eq, Eq):
#             lhs, rhs = eq.lhs, eq.rhs
#         else:
#             lhs, rhs = eq[0], eq[1]  # in case it's a tuple (lhs, rhs)
#         all_symbols.update(lhs.free_symbols)
#         all_symbols.update(rhs.free_symbols)

#     # 2. Identify variable names that are exogenous
#     known_names = set(state.keys())

#     # 3. Initialize them with random values (e.g. between 0 and 1)
#     new_state = state.copy()
#     for sym in all_symbols:
#         name = str(sym)
#         if name not in known_names:
#             new_state[name] = random.uniform(0.1, 0.5)

#     return new_state


class SFCConsistencyAdjuster:
    
    def __init__(self, balance_sheet_map, equations=None):
        """
        Initialize with balance sheet map and optional behavioral equations.
        
        Parameters:
        - balance_sheet_map: dict mapping variable names to (instrument, sector, sign)
        - equations: list of sympy Eq objects (behavioral equations)
        """
        self.balance_sheet_map = balance_sheet_map
        self.equations = equations if equations else []
        
        # Extract all sectors and instruments from balance sheet map
        self.instruments = list(set([v[0] for v in balance_sheet_map.values()]))
        self.sectors = list(set([v[1] for v in balance_sheet_map.values()]))
        
    def build_matrix(self, state):
        return build_matrix(state)
    
    def check_consistency(self, matrix):
        return check_matrix_consistency(matrix, tol=1e-4, verbose=True)
    
    def apply_adjustments(self, state, matrix, differences, relaxation=0.5):
        """Apply adjustments to reduce inconsistencies."""
        state_adjusted = state.copy()
        
        # 1. ADJUST FOR ROW IMBALANCES
        row_errors = differences['row_differences']
        
        for instrument, error in row_errors.items():
            if abs(error) < 1e-8:
                continue
                
            # Find variables affecting this instrument
            relevant_vars = []
            for var_name, (instr, sector, sign) in self.balance_sheet_map.items():
                if instr == instrument:
                    clean_var = re.sub(r"\(.*?\)", "", var_name)
                    if "*" in clean_var:
                        factors = clean_var.split("*")
                        # Adjust the stock component, not the price
                        stock_var = factors[1] if factors[1] in state else factors[0]
                        if stock_var in state:
                            relevant_vars.append((stock_var, sign, sector))
                    else:
                        if clean_var in state:
                            relevant_vars.append((clean_var, sign, sector))
            
            if not relevant_vars:
                continue
                
            # Distribute adjustment proportionally
            total_abs = sum(abs(state[var]) for var, _, _ in relevant_vars)
            if total_abs > 0:
                for var_name, sign, sector in relevant_vars:
                    weight = abs(state[var_name]) / total_abs
                    adjustment = -error * weight * relaxation * sign
                    state_adjusted[var_name] = state[var_name] + adjustment
        
        # 2. ADJUST FOR COLUMN IMBALANCES
        col_errors = differences['column_differences']
        
        # Special handling for Bank Capital (appears in two places)
        if 'Bank Capital' in matrix.index:
            bank_capital_vars = []
            for var_name, (instr, sector, sign) in self.balance_sheet_map.items():
                if instr == 'Bank Capital':
                    clean_var = re.sub(r"\(.*?\)", "", var_name)
                    if clean_var in state:
                        bank_capital_vars.append((clean_var, sign, sector))
            
            if len(bank_capital_vars) == 2:
                var1, sign1, _ = bank_capital_vars[0]
                var2, sign2, _ = bank_capital_vars[1]
                
                current_sum = sign1 * state[var1] + sign2 * state[var2]
                if abs(current_sum) > 1e-8:
                    # Adjust to make sum zero
                    factor = 0 / current_sum if current_sum != 0 else 1
                    state_adjusted[var1] = state[var1] * factor
                    state_adjusted[var2] = state[var2] * factor
        
        # 3. ENFORCE WEALTH CONSERVATION
        wealth_error = differences['wealth_discrepancy']
        
        if abs(wealth_error) > 1e-8:
            # Adjust household wealth
            if 'V_-1' in state_adjusted:
                state_adjusted['V_-1'] = state['V_-1'] + wealth_error * relaxation
        
        return state_adjusted
    
    def initialize_state(self, initial_guess=None, target_tol=1e-4, 
                         max_iterations=1000, relaxation=0.5, # adjustment speed
                         verbose=False, random_seed=None):

        if random_seed is not None:
            np.random.seed(random_seed)
        
        # If no initial guess, create random starting point
        if initial_guess is None:
            initial_guess = self._create_random_state(seed=random_seed)
        
        working_state = initial_guess.copy()
        history = []
        
        for iteration in range(max_iterations):
            # Build and check matrix
            matrix = self.build_matrix(working_state)
            is_consistent, differences = self.check_consistency(
                matrix, tol=target_tol
            )
            
            # Track progress
            max_error = max(
                abs(differences['row_differences']).max() if len(differences['row_differences']) > 0 else 0,
                abs(differences['column_differences']).max(),
                abs(differences['wealth_discrepancy'])
            )
            history.append(max_error)
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Max error = {max_error:.6f}")
            
            # Check convergence
            if is_consistent:
                if verbose:
                    print(f"\n✓ Converged after {iteration} iterations")
                    self._print_consistency_report(matrix, differences)
                return working_state, matrix, (True, differences)
            
            # Apply adjustments
            working_state = self.apply_adjustments(
                working_state, matrix, differences, relaxation
            )
            
            # Check for divergence
            if iteration > 10 and history[-1] > history[-10] * 1.5:
                if verbose:
                    print(f"Warning: Errors increasing, reducing relaxation")
                relaxation *= 0.5
        
        if verbose:
            print(f"\n✗ Failed to converge after {max_iterations} iterations")
            self._print_consistency_report(matrix, differences)
        
        return working_state, matrix, (False, differences)
    
    def _create_random_state(self, base_state=None, seed=None):
        """Create a random initial state by perturbing base values."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Use provided base or create empty
        if base_state is None:
            base_state = {}
        
        random_state = base_state.copy()
        
        # Add random values for variables in balance sheet map
        all_vars = set()
        for var_name in self.balance_sheet_map.keys():
            clean_var = re.sub(r"\(.*?\)", "", var_name)
            if "*" in clean_var:
                factors = clean_var.split("*")
                all_vars.update(factors)
            else:
                all_vars.add(clean_var)
        
        for var in all_vars:
            if var not in random_state:
                # Random value between 1000 and 1000000 for stocks
                if var.endswith('-1'):
                    random_state[var] = np.random.uniform(1000, 1000000)
                else:
                    # Parameters between 0 and 1
                    random_state[var] = np.random.uniform(0, 1)
        
        return random_state


# Modified initialize_variables function that uses the adjuster
def initialize_variables(equations, state, seed=0, balance_sheet_map=None):

    random.seed(seed)
    np.random.seed(seed)
    
    if balance_sheet_map is None:
        # Try to import from global scope
        try:
            from __main__ import balance_sheet_map
        except:
            raise ValueError("balance_sheet_map must be provided")
    
    # Step 1: Collect all symbols from equations
    all_symbols = set()
    for eq in equations:
        if isinstance(eq, Eq):
            all_symbols.update(eq.lhs.free_symbols)
            all_symbols.update(eq.rhs.free_symbols)
    
    # Step 2: Create initial random state for unknown variables
    known_names = set(state.keys())
    new_state = state.copy()
    
    for sym in all_symbols:
        name = str(sym)
        if name not in known_names:
            # Different ranges for different types
            if name.endswith('-1'):  # Stock variables
                new_state[name] = random.uniform(1000, 1000000)
            elif name.startswith('p_'):  # Prices
                new_state[name] = random.uniform(0.5, 20)
            else:  # Parameters and flows
                new_state[name] = random.uniform(0, 1)
    
    # Step 3: Use adjuster to enforce SFC constraints
    adjuster = SFCConsistencyAdjuster(balance_sheet_map, equations)
    
    consistent_state, matrix, (success, _) = adjuster.initialize_state(
        initial_guess=new_state,
        verbose=True,
        max_iterations=1000,
        random_seed=seed
    )
    
    if success:
        print("✓ Successfully initialized consistent state")
    else:
        print("⚠ Warning: State may not be fully consistent")
    
    return consistent_state

