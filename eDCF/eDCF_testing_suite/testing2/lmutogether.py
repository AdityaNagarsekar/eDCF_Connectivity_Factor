import math
from functools import lru_cache
import pandas as pd
import matplotlib.pyplot as plt

# --- Function Definitions ---

def lcf(m, n):
    """Calculates the Lower Bound (LCF)."""
    if n == 0:
        return 1.0 if m == 0 else 0.0
    
    temp = 7 ** m - 3 ** m
    temp1 = (3 ** n - 1) * (3 ** m)
    
    if temp1 == 0:
        return float('inf') 
    return temp / temp1
    
def mcf(m, n):
    """Calculates the Middle Bound (MCF)."""
    if n == 0:
        return 0.0
        
    temp = 3 ** m - 1
    temp1 = 3 ** n - 1

    return temp / temp1

@lru_cache(maxsize=None)
def calculate_alpha(x: int, m: int, n: int) -> float:
    """
    Implements the FINAL MODIFIED formula (A):
    1. 2^x is replaced by 2^i inside the summation.
    2. Summation for i starts from 0 instead of 1.
    """
    sum_val = 0
    for i in range(x + 1): 
        k = m - (x - i)
        if 0 <= k <= n - x:
            term = (
                (2**i) *
                math.comb(x, i) *
                (2**k) *
                math.comb(n - x, k)
            )
            sum_val += term
            
    total_sum = sum_val
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
    # 1. Generate data
    print("Calculating values...")
    results = []
    for n in range(11):
        for m in range(n + 1):
            if m == 0:
               calculate_alpha.cache_clear()

            results.append({
                'm': m, 
                'n': n, 
                'lcf': lcf(m, n), 
                'mcf': mcf(m, n), 
                'ucf': ucf(m, n)
            })
    
    # 2. Create DataFrame
    df = pd.DataFrame(results)
    
    # 3. Save data to CSV
    df.to_csv('lcf_mcf_ucf_results_final.csv', index=False)
    print("Saved the results to 'lcf_mcf_ucf_results_final.csv'")

    # 4. Create and save the separate plots
    
    # Plot LCF
    fig_lcf, ax_lcf = plt.subplots(figsize=(11, 8))
    for n_val, group in df.groupby('n'):
        ax_lcf.plot(group['m'], group['lcf'], marker='o', linestyle='-', label=f'n = {n_val}')
    ax_lcf.set_title('Lower Bound (LCF) vs. m for different n')
    ax_lcf.set_ylabel('LCF Value')
    ax_lcf.set_xlabel('m')
    ax_lcf.legend(title='n', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    ax_lcf.grid(True)
    fig_lcf.tight_layout(rect=[0, 0, 0.85, 1])
    fig_lcf.savefig('lcf_plot.png')
    print("Saved the LCF plot to 'lcf_plot.png'")

    # Plot MCF
    fig_mcf, ax_mcf = plt.subplots(figsize=(11, 8))
    for n_val, group in df.groupby('n'):
        ax_mcf.plot(group['m'], group['mcf'], marker='o', linestyle='-', label=f'n = {n_val}')
    ax_mcf.set_title('Middle Bound (MCF) vs. m for different n')
    ax_mcf.set_ylabel('MCF Value')
    ax_mcf.set_xlabel('m')
    ax_mcf.legend(title='n', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    ax_mcf.grid(True)
    fig_mcf.tight_layout(rect=[0, 0, 0.85, 1])
    fig_mcf.savefig('mcf_plot.png')
    print("Saved the MCF plot to 'mcf_plot.png'")

    # Plot UCF
    fig_ucf, ax_ucf = plt.subplots(figsize=(11, 8))
    for n_val, group in df.groupby('n'):
        ax_ucf.plot(group['m'], group['ucf'], marker='o', linestyle='-', label=f'n = {n_val}')
    ax_ucf.set_title('Upper Bound (UCF) vs. m for different n')
    ax_ucf.set_ylabel('UCF Value')
    ax_ucf.set_xlabel('m')
    ax_ucf.legend(title='n', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    ax_ucf.grid(True)
    fig_ucf.tight_layout(rect=[0, 0, 0.85, 1])
    fig_ucf.savefig('ucf_plot.png')
    print("Saved the UCF plot to 'ucf_plot.png'")