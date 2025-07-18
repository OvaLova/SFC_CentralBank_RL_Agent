import pandas as pd
import re


balance_sheet_map = {
    "IN_-1": ("Inventories", "Firms", +1),
    "K_-1": ("Fixed capital", "Firms", +1),
    "H_s-1": ("Cash", "CentralBank", -1),
    "H_hd-1": ("Cash", "Households", +1),
    "H_bd-1": ("Cash", "Banks", +1),
    "M_s-1": ("Money Deposits", "Banks", -1),
    "M_d-1": ("Money Deposits", "Households", +1),
    "B_s-1": ("Bills", "Government", -1),
    "B_hd-1": ("Bills", "Households", +1),
    "B_cbd-1": ("Bills", "CentralBank", +1),
    "B_bd-1": ("Bills", "Banks", +1),
    "p_bL-1*BL_d-1": ("Bonds", "Households", +1),
    "p_bL-1*BL_s-1": ("Bonds", "Government", -1),
    "L_s-1": ("Loans", "Banks", +1),
    "L_fd-1": ("Loans", "Firms", -1),
    "L_hd-1": ("Loans", "Households", -1),
    "p_e-1*e_d-1": ("Equities", "Households", +1),
    "p_e-1*e_s-1": ("Equities", "Firms", -1),
    "OF_b-1(banks)": ("Bank Capital", "Banks", -1),
    "OF_b-1(households)": ("Bank Capital", "Households", +1),
    # Balancing items
    "V_-1": ("Wealth", "Households", -1),  
    "GD_-1": ("Wealth", "Government", +1), 
    "V_f-1": ("Wealth", "Firms", -1)
}

state = {
    # Starting values for parameters
    "α_1": 0.75,
    "α_2": 0.064,
    "β": 0.5,
    "β_b": 0.4,
    "γ": 0.15,
    "gr_0": 0.00122,
    "γ_r": 0.1,
    "γ_u": 0.05,
    "δ": 0.10667,
    "δ_rep": 0.1,
    "ε": 0.5,
    "ε_2": 0.8,
    "ε_b": 0.25,
    "η": 0.04918,
    "η_0": 0.07416,
    "η_n": 0.6,
    "η_r": 0.4,
    "θ": 0.22844,
    "λ_20": 0.25,
    "λ_21": 2.2,
    "λ_22": 6.6,
    "λ_23": 2.2,
    "λ_24": 2.2,
    "λ_25": 0.1,
    "λ_30": -0.04341,
    "λ_31": 2.2,
    "λ_32": 2.2,
    "λ_33": 6.6,
    "λ_34": 2.2,
    "λ_35": 0.1,
    "λ_40": 0.67132,
    "λ_41": 2.2,
    "λ_42": 2.2,
    "λ_43": 2.2,
    "λ_44": 6.6,
    "λ_45": 0.1,
    "λ_b": 0.0153,
    "λ_c": 0.05,
    "ζ_m": 0.00075,
    "ζ_m1": 0.0008,
    "ζ_m2": 0.0007,
    "ρ": 0.05,
    "σ^N": 0.1666,
    "σ_se-1": 0.16667,
    "σ^T": 0.2,
    "φ_-1": 0.26417,
    "φ^T_-1": 0.26417,
    "ψ_D": 0.15255,
    "ψ_U": 0.92,
    "Ω_0": -0.20594,
    "Ω_1": 1,
    "Ω_2": 2,
    "Ω_3": 0.45621,
    # Values for exogenous variables
    "add_bL": 0.02,
    "bandT": 0.01,
    "bandB": 0.01,
    "bot": 0.05,
    "gr_g": 0.03,
    "gr_pr": 0.03,
    "N_fe-1": 87.181,
    "NCAR": 0.1,
    "npl": 0.02,
    "r_b_": 0.035,
    "r_lN": 0.07,
    "top": 0.12,
    # Initial values for stocks
    "B_bs-1": 4389790,
    "B_bd-1": 4389790,
    "B_cbd-1": 4655690,
    "B_cbs-1": 4655690,
    "B_hd-1": 33439320,
    "B_hs-1": 33439320,
    "B_s-1": 42484800,
    "BL_d-1": 840742,
    "BL_s-1": 840742,
    # "GD_-1": 57728700, # original
    "GD_-1": 33439320 + 4389790 + 840742*18.182 + 2025540 + 2630150, # replacement
    "e_d-1": 5112.6001,
    "e_s-1": 5112.6001,
    "H_bd-1": 2025540,
    "H_bs-1": 2025540,
    "H_hd-1": 2630150,
    "H_hs-1": 2630150,
    "H_s-1": 2025540 + 2630150,
    "IN_-1": 11585400,
    "in_-1": 2064890,
    "in^e_-1": 2405660,
    "in^T_-1": 2064890,
    "K_-1": 127444000,
    "k_-1": 17768900,
    "L_fd-1": 15962900,
    "L_fs-1": 15962900,
    "L_hd-1": 21606600,
    "L_hs-1": 21606600,
    "L_s-1": 15962900 + 21606600,
    "M_d-1": 40510800,
    "M_s-1": 40510800,
    "OF_b-1": 3474030,
    "OF^e_b-1": 3474030,
    "OF^T_b-1": 3638100,
    # "V_-1": 165438779, # original
    "V_-1": 165438779 + 0.0377, # replacement
    "V_f-1": 11585400 + 127444000 - 15962900 - 5112.6001*17937, # added
    "V_fma-1": 159334599,
    "v_-1": 165438779/7.1723,
    # Initial values for other endogenous
    "add_l": 0.04592,
    "BLR_-1": 0.1091,
    "BUR_-1": 0.06324,
    "c_-1": 7334240,
    "CAR_-1": 0.09245,
    "C_-1": 52603100,
    "ER_-1": 1,
    "F_b-1": 1744130,
    "F^T_b-1": 1744140,
    "F_f-1": 18081100,
    "F^T_f-1": 18013600,
    "FD_b-1": 1325090,
    "FD_f-1": 2670970,
    "FU_b-1": 419039,
    "FU_f-1": 15153800,
    "FU^T_f-1": 15066200,
    "G_-1": 16755600,
    "g_-1": 2336160,
    "GL_-1": 2775900,
    "gr_k_-1": 0.03001,
    "I_-1": 16911600,
    "i_-1": 2357910,
    "N_-1": 87.181,
    "N^T_-1": 87.181,
    "NHUC_-1": 5.6735,
    "NL_-1": 683593,
    "nl_-1": 95311,
    "NPL_-1": 309158,
    "npl^e_-1": 0.02,
    "NUC_-1": 5.6106,
    "ω^T_-1": 112852,
    "p_-1": 7.1723,
    "p_bL-1": 18.182,
    "p_e-1": 17937,
    "PE_-1": 5.07185,
    "π_-1": 0.0026,
    "pr_-1": 138659,
    "PSBR_-1": 1894780,
    "q_-1": 0.77443,
    "r_b-1": 0.035,
    "r_bL-1": 0.055,
    "r_K-1": 0.03008,
    "r_l-1": 0.06522,
    "r_m-1": 0.0193,
    "REP_-1": 2092310,
    "rr_b-1": 0.03232,
    "rr_l-1": 0.06246,
    "S_-1": 86270300,
    "s_-1": 12028300,
    "s^e_-1": 12028300,
    "T_-1": 17024100,
    "u_-1": 0.70073,
    "UC_-1": 5.6106,
    "W_-1": 777968,
    "WB_-1": 67824000,
    "Y_-1": 86607700,
    "y_-1": 12088400,
    "YD_r-1": 56446400,
    "yd_r-1": 7813270,
    "yd^e_r-1": 7813290,
    "YP_-1": 73158700,
    # Missing
    # "HC^e_-1": 5.6106*12028300*(1-0.16667) + 5.6106*12028300*0.16667*(0.06522+1),
    # "YD_hs-1": 56446400 + 0,
    # "CG_-1": 0,
    # "F_cb-1": 0.035*4655690,
    # "FU^T_b-1": 419039,
}

def build_matrix(var_map, state):
    instruments = list(set([v[0] for v in var_map.values()]))
    sectors = list(set([v[1] for v in var_map.values()]))
    matrix = pd.DataFrame(0.0, index=instruments, columns=sectors)

    for varname, (instr, sector, sign) in var_map.items():
        clean_var = re.sub(r"\(.*?\)", "", varname)
        if "*" in clean_var:
            factors = clean_var.split("*")
            value = 1.0
            for f in factors:
                if f in state:
                    value *= state[f]
                else:
                    value = None
                    break
            if value is not None:
                matrix.loc[instr, sector] += sign * value
        else:
            if clean_var in state:
                matrix.loc[instr, sector] += sign * state[clean_var]
    return matrix

def check_matrix_consistency(matrix, tol=1e-4):
    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)
    excluded_rows = ['Inventories', 'Fixed capital', 'Wealth']
    filtered_rows = row_sums[~row_sums.index.isin(excluded_rows)]
    row_consistency = filtered_rows.abs().max() < tol
    col_consistency = column_sums.abs().max() < tol
    total_wealth = -matrix.loc['Wealth'].sum()
    total_real_assets = (
        matrix.loc['Inventories'].sum() + 
        matrix.loc['Fixed capital'].sum()
    )
    wealth_consistency = abs(total_wealth - total_real_assets) < tol
    # Print diagnostic information
    print("\n=== Accounting Consistency Check ===")
    print(f"Row consistency: {'PASS' if row_consistency else 'FAIL'}")
    print(f"Column consistency: {'PASS' if col_consistency else 'FAIL'}")
    print(f"Wealth Conservation: {'PASS' if wealth_consistency else 'FAIL'}")
    if not row_consistency:
        print("\nProblematic instrument sums:")
        print(filtered_rows[filtered_rows.abs() >= tol])
    if not col_consistency:
        print("\nProblematic sector sums:")
        print(column_sums[column_sums.abs() >= tol])
    if not wealth_consistency:
        print("\nWealth conservation broken:")
        print(f"  Difference: {total_wealth - total_real_assets}")
    return row_sums, column_sums

def balance_sector(matrix, sector, balancing_instrument="Wealth"):
    imbalance = matrix[sector].sum()
    matrix.loc[balancing_instrument, sector] -= imbalance

