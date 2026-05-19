# Implied Volatility Calibration

Quantitative finance project focused on implied volatility calibration using the Black-Scholes model and market option data.

## Project Overview

This project computes implied volatilities from market option prices and visualizes volatility smiles for different maturities.

The project includes:
- Black-Scholes pricing
- Newton-Raphson implied volatility calibration
- Call and Put option pricing
- Dividend-adjusted pricing
- Volatility smile visualization
- 2D and 3D volatility plots
- Linear regression for dividend yield estimation

---

## Methods

### Black-Scholes Model

The implied volatility is obtained by solving:

```text
Black-Scholes price = Market option price
```

using the Newton-Raphson algorithm.

### Dividend Adjustment

For dividend-paying assets, the underlying price is adjusted using:

```text
S0 * exp(-qT)
```

where `q` is the dividend yield.

### Linear Regression

The project estimates:
- the initial underlying price,
- the dividend yield,

using call-put parity and linear regression techniques.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Black-Scholes model
- Newton-Raphson method
- Linear regression

---

## Repository Structure

```text
.
├── TP1.py
├── Calibration de volatilité implicite.pdf
├── TP1_ING3_Calibration_2025_26.pdf
├── GoogleOrig.xlsx
├── NEW_spx_quotedata.xlsx
├── sp-index.txt
├── spx_quotedata.xlsx
├── README.md
└── LICENSE
```

---

## How to Run

Install the required packages:

```bash
pip install pandas numpy matplotlib openpyxl
```

Run the script:

```bash
python TP1.py
```

---

## Main Features

- Implied volatility extraction
- Volatility smile generation
- 2D and 3D smile visualization
- Call/Put volatility analysis
- Dividend yield estimation
- Market data processing

---

## Financial Concepts

- Implied volatility
- Volatility smiles
- Black-Scholes pricing
- Option calibration
- Dividend-adjusted pricing
- Numerical optimization

---

## Results

The project generates:
- implied volatility smiles,
- volatility surfaces,
- dividend yield estimates,
- regression visualizations,
- market-consistent volatility structures.

The implementation uses real option market datasets and numerical calibration methods. :contentReference[oaicite:0]{index=0}

---

## Author

Bertrand Lecoeur  
CY Tech – Financial Engineering
