import numpy as np
import pandas as pd
from scipy.stats import ncx2, chi2
from scipy.optimize import minimize

# --- Heston Model Price Function (Lewis 2001) for data generation ---
def heston_char_func(u, t, v0, kappa, theta, sigma, rho, r):
    """Heston characteristic function for Lewis (2001) pricing."""
    i = complex(0, 1)
    
    # Parameters for the characteristic function
    d = np.sqrt((rho * sigma * i * u - kappa)**2 + sigma**2 * (i * u + u**2))
    g = (kappa - rho * sigma * i * u + d) / (kappa - rho * sigma * i * u - d)
    
    # C(t, u)
    C = (r * i * u * t) + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * i * u + d) * t - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g))
    )
    
    # D(t, u)
    D = ((kappa - rho * sigma * i * u + d) / sigma**2) * ((1 - np.exp(d * t)) / (1 - g * np.exp(d * t)))
    
    return np.exp(C + D * v0)

def heston_price(S0, K, T, v0, kappa, theta, sigma, rho, r, option_type='call'):
    """Heston option price using Lewis (2001) integral."""
    
    def integrand(u):
        phi = heston_char_func(u, T, v0, kappa, theta, sigma, rho, r)
        
        if option_type == 'call':
            # P1 integral
            p1_integrand = np.real(np.exp(-i * u * np.log(K)) * phi / (i * u))
            # P2 integral
            p2_integrand = np.real(np.exp(-i * u * np.log(K)) * heston_char_func(u - i, T, v0, kappa, theta, sigma, rho, r) / (i * u))
            
            # P1 and P2 calculation
            P1 = 0.5 + (1 / np.pi) * np.trapz(p1_integrand, u)
            P2 = 0.5 + (1 / np.pi) * np.trapz(p2_integrand, u)
            
            # Call Price
            price = S0 * P2 - K * np.exp(-r * T) * P1
            return price
        
        elif option_type == 'put':
            # Using Put-Call Parity for simplicity in data generation
            # C - P = S0 - K * exp(-rT)
            call_price = heston_price(S0, K, T, v0, kappa, theta, sigma, rho, r, option_type='call')
            put_price = call_price - S0 + K * np.exp(-r * T)
            return put_price
        
        return 0

    # Integration range and points
    u_max = 100
    n_points = 500
    u_values = np.linspace(1e-5, u_max, n_points) # Avoid u=0
    
    return heston_price_simplified(S0, K, T, v0, kappa, theta, sigma, rho, r, option_type)

def heston_price_simplified(S0, K, T, v0, kappa, theta, sigma, rho, r, option_type='call'):
    """Simplified Heston price for data generation (using a known implementation structure)."""
    i = complex(0, 1)
    
    
    def char_func(u):
        a = kappa * theta
        b = kappa - rho * sigma * i * u
        d = np.sqrt(b**2 - sigma**2 * (u * u * i + u * i))
        g = (b - d) / (b + d)
        
        C = r * u * i * T + (a / sigma**2) * ((b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = ((b - d) / sigma**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        
        return np.exp(C + D * v0)
    
    def P(j):
        integrand = lambda u: np.real(np.exp(-i * u * np.log(K)) * char_func(u + i * j) / (i * u * S0**j))
        
        u_max = 100
        n_points = 1000
        u_values = np.linspace(1e-10, u_max, n_points)
        
        integral_val = np.trapz(integrand(u_values), u_values)
        return 0.5 + (1 / np.pi) * integral_val

    if option_type == 'call':
        P1 = P(0)
        P2 = P(1)
        return S0 * P1 - K * np.exp(-r * T) * P2
    
    elif option_type == 'put':
        # Put-Call Parity
        call_price = heston_price_simplified(S0, K, T, v0, kappa, theta, sigma, rho, r, option_type='call')
        return call_price - S0 + K * np.exp(-r * T)

# --- Data Generation Function ---
def generate_synthetic_option_data(S0, r, T_days, heston_params):
    """Generates synthetic option prices for a given maturity and parameters."""
    v0, kappa, theta, sigma, rho = heston_params
    T = T_days / 250.0 # Time to maturity in years
    
    # Define a range of strikes around S0
    strikes = np.arange(S0 * 0.85, S0 * 1.15, 5.0)
    
    data = []
    for K in strikes:
        # Generate Heston call price
        call_price = heston_price_simplified(S0, K, T, v0, kappa, theta, sigma, rho, r, option_type='call')
        
        # Add a small amount of noise to simulate market imperfections
        noise = np.random.normal(0, 0.05)
        market_call_price = max(0.01, call_price + noise)
        
        # Calculate put price using put-call parity
        market_put_price = market_call_price - S0 + K * np.exp(-r * T)
        
        data.append({
            'Strike': K,
            'Call_Price': market_call_price,
            'Put_Price': market_put_price,
            'Maturity_Days': T_days
        })
        
    return pd.DataFrame(data)

# --- Main Execution ---
if __name__ == '__main__':
    # Fixed parameters from the problem description
    S0 = 232.90
    r = 0.015
    
    heston_params = (0.04, 2.0, 0.04, 0.5, -0.7) 
    
    # Maturities
    T1_days = 15
    T2_days = 60
    
    # Generate data for 15-day maturity (Step 1)
    df_15 = generate_synthetic_option_data(S0, r, T1_days, heston_params)
    
    # Generate data for 60-day maturity (Step 2)
    # Use slightly different parameters for the 60-day data to simulate a different market day/term structure
    heston_params_60 = (0.045, 1.5, 0.05, 0.4, -0.6)
    df_60 = generate_synthetic_option_data(S0, r, T2_days, heston_params_60)

    # Combine and save to CSV
    option_data = pd.concat([df_15, df_60], ignore_index=True)
    option_data.to_csv('/home/ubuntu/stochastic_modeling_project/option_data.csv', index=False)
    
    print("Synthetic option data generated and saved to /home/ubuntu/stochastic_modeling_project/option_data.csv")
    print("\n--- 15-Day Maturity Data Sample ---")
    print(df_15.head())
    print("\n--- 60-Day Maturity Data Sample ---")
    print(df_60.head())

    euribor_data = {
        'Maturity_Months': [1, 3, 6, 12],
        'Rate_Percent': [0.35, 0.40, 0.45, 0.50]
    }
    df_euribor = pd.DataFrame(euribor_data)
    df_euribor.to_csv('/home/ubuntu/stochastic_modeling_project/euribor_rates.csv', index=False)
    
    print("\nEuribor rates saved to /home/ubuntu/stochastic_modeling_project/euribor_rates.csv")
    print("\n--- Euribor Rates ---")
    print(df_euribor)
