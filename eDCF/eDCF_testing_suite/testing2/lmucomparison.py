import math
from functools import lru_cache
import pandas as pd
import matplotlib.pyplot as plt

# --- Function Definitions ---

def lcf(m, n):
    """Calculates the Lower Bound (LCF)."""
    if n == 0:
        return 1.0 if m == 0 else 1.0 # Handle division by zero and (0,0) case
    
    temp = 7 ** m - 3 ** m
    temp1 = (3 ** n - 1) * (3 ** m)
    
    if temp1 == 0:
        return float('inf') 
    return temp / temp1
    
def mcf(m, n):
    """Calculates the Middle Bound (MCF)."""
    if n == 0:
        # Denominator is zero, and for a valid pair, m must also be 0. 
        # The result is 0/0, which we can define as 0.
        return 1
        
    temp = 3 ** m - 1
    temp1 = 3 ** n - 1

    return temp / temp1

# --- FINAL MODIFIED Alpha Function ---
@lru_cache(maxsize=None)
def calculate_alpha(x: int, m: int, n: int) -> float:
    """
    Implements the FINAL MODIFIED formula (A):
    1. 2^x is replaced by 2^i inside the summation.
    2. Summation for i starts from 0 instead of 1.
    """
    sum_val = 0
    # *** CHANGE IS HERE: Summation for i now starts from 0 ***
    for i in range(x + 1): 
        k = m - (x - i)
        # The term is zero if k is out of bounds for the binomial coefficient
        if 0 <= k <= n - x:
            term = (
                (2**i) *
                math.comb(x, i) *
                (2**k) *
                math.comb(n - x, k)
            )
            sum_val += term
            
    total_sum = sum_val
    
    # Subtract the Kronecker delta term
    kronecker_delta = 1 if x == m else 0
    
    return float(total_sum - kronecker_delta)

def calculate_m_chi(t: int, m: int, n: int) -> float:
    """Implements formula (B) from the original problem image."""
    if m > n:
        return 0.0
        
    numerator_sum = 0
    for i in range(n - m, n + 1):
        numerator_sum += calculate_alpha(x=t, m=i, n=n)
        
    denominator = (3**n) - 1
    if denominator == 0:
        return 0.0
        
    return numerator_sum / denominator

def ucf(m: int, n: int) -> float:
    """
    Implements formulas (C) and (D) to calculate the Upper Bound (UCF).
    """
    if n == 0 and m == 0:
        return 1.0
    
    if not (0 <= m <= n):
        raise ValueError("Inputs must satisfy 0 <= m <= n.")

    f_t_denominator = sum(math.comb(n, u) for u in range(n - m, n + 1))
    
    if f_t_denominator == 0:
        return 0.0

    ucf_sum = 0.0
    for t in range(n - m, n + 1):
        f_t_numerator = math.comb(n, t)
        f_t = f_t_numerator / f_t_denominator
        
        m_chi_t = calculate_m_chi(t, m, n)
        
        ucf_sum += f_t * m_chi_t
        
    return ucf_sum

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Set the list of n values to be plotted.
    N_VALUES = [1, 5, 10, 20, 30, 50]

    for target_n in N_VALUES:
        print(f"--- Processing for n = {target_n} ---")
        # 2. Generate data for the target n
        print(f"Calculating values for LCF, MCF, and UCF...")
        results = []
        calculate_alpha.cache_clear() # Clear cache for each new n
        for m in range(target_n + 1):
            lcf_val = lcf(m, target_n)
            mcf_val = mcf(m, target_n)
            ucf_val = ucf(m, target_n)
            results.append({
                'm': m,
                'n': target_n,
                'lcf': lcf_val,
                'mcf': mcf_val,
                'ucf': ucf_val
            })

        # 3. Create a DataFrame from the results
        df = pd.DataFrame(results)

        # 4. Save the data to a CSV file (same as before)
        csv_filename = f'lcf_mcf_ucf_results_n_{target_n}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Saved the results to '{csv_filename}'")

        # 5. Create and save the plot with a logarithmic y-axis
        plt.figure(figsize=(12, 8)) # Create a new figure for each plot
        plt.plot(df['m'], df['lcf'], marker='o', linestyle='-', label='LCF')
        plt.plot(df['m'], df['mcf'], marker='s', linestyle='--', label='MCF')
        plt.plot(df['m'], df['ucf'], marker='^', linestyle='-.', label='UCF')

        plt.title(f'Comparison of LCF, MCF, and UCF for n = {target_n} (Log Scale)')
        plt.xlabel('m')
        plt.ylabel('Value (Symmetrical Log Scale)')
        
        # --- CHANGE IS HERE: Set y-axis to a symmetrical log scale ---
        # This handles the zero values gracefully
        plt.yscale('symlog', linthresh=1e-5) 
        
        plt.legend(fontsize=20)
        plt.grid(True, which='both')
        plt.tight_layout()

        # Update filename to reflect the log scale
        plot_filename = f'lcf_mcf_ucf_plot_log_n_{target_n}.png'
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free up memory
        print(f"Saved the log scale plot to '{plot_filename}'")
    print("\n--- All plots and data files generated successfully! ---")