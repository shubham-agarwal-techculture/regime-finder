# Regime Finder

A machine learning-based financial market regime detection tool that identifies different market states using K-means clustering on momentum and volatility features. This tool analyzes NIFTY 50 stock market data and builds a probabilistic model of market regimes for simulation and forecasting.

## Overview

The Regime Finder uses unsupervised learning to detect distinct market regimes characterized by different momentum and volatility patterns. By clustering historical market data, it identifies different market states and learns the probability of transitions between them, enabling market regime awareness for trading strategies and risk management.

## Features

- **Data Fetching**: Connects to QuestDB database to fetch NIFTY 50 OHLC (Open, High, Low, Close) data
- **Feature Engineering**: Computes multi-horizon momentum and volatility indicators
  - Momentum: 5-day, 20-day, and 60-day returns
  - Volatility: 5-day, 20-day, and 60-day rolling standard deviations
- **Regime Detection**: Uses K-means clustering to identify distinct market regimes
- **State Transition Analysis**: Builds transition probability matrices between states
- **Distribution Fitting**: Models return distributions within each regime
- **Market Simulation**: Generates synthetic market paths based on learned regime dynamics

## Installation

### Prerequisites
- Python 3.12 or higher
- QuestDB instance with NIFTY 50 data

### Setup

1. Clone or download the repository:
```bash
cd regime_finder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, using `pyproject.toml`:
```bash
pip install -e .
```

## Dependencies

- **numpy** (≥2.4.4): Numerical computations
- **pandas** (≥3.0.2): Data manipulation and analysis
- **scikit-learn** (≥1.8.0): K-means clustering and scaling
- **psycopg2** (≥2.9.11): PostgreSQL database adapter
- **asyncpg** (≥0.31.0): Async PostgreSQL driver
- **sqlalchemy** (≥2.0.48): SQL toolkit and ORM

## Configuration

Update the database connection parameters in `main.py`:

```python
engine = create_engine("postgresql+psycopg2://admin:quest@192.168.1.101:8812/qdb")
```

Modify the following as needed:
- **Host**: Database server IP (default: 192.168.1.101)
- **Port**: QuestDB PostgreSQL port (default: 8812)
- **User**: Database username (default: admin)
- **Password**: Database password (default: quest)
- **Database**: Database name (default: qdb)

## Usage

Run the main script:
```bash
python main.py
```

### Core Functions

- **`fetch_questdb_data()`**: Fetches NIFTY 50 OHLC data from QuestDB
- **`compute_features(df, price_col="close")`**: Calculates momentum and volatility features
- **`cluster_states(df, n_states=5)`**: Performs K-means clustering to identify regimes
- **`compute_transition_matrix(states, n_states)`**: Builds state transition probability matrix
- **`fit_state_distributions(df)`**: Fits return distributions for each state (Gaussian)
- **`simulate_market(n_steps, transition_matrix, state_models, start_state=0)`**: Generates synthetic market paths
- **`build_model(df, n_states=5)`**: End-to-end pipeline to build the regime model

## How It Works

1. **Data Collection**: NIFTY 50 price data is fetched from QuestDB
2. **Feature Engineering**: Computes momentum (returns at different horizons) and volatility indicators
3. **Standardization**: Features are scaled using StandardScaler for clustering
4. **Regime Clustering**: K-means identifies n_states distinct market regimes
5. **Transition Modeling**: Learns empirical probability of moving from one regime to another
6. **Return Distribution**: Models return distribution within each regime as Gaussian
7. **Simulation**: Generates synthetic market paths by sampling states and returns

## Flow Architecture

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      REGIME FINDER SYSTEM                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  QuestDB Server  │
                    │  (NIFTY 50 Data) │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ fetch_questdb_data() │
                    │  (Load OHLC Data)    │
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────────┐
                    │  compute_features()      │
                    │  - Returns (1, 5, 20, 60)│
                    │  - Volatility (5, 20, 60)│
                    └──────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────────┐
                    │   StandardScaler         │
                    │  (Normalize Features)    │
                    └──────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 REGIME DETECTION BLOCK                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │  cluster_states()│─────▶│  K-Means Model   │             │
│  │  (n_states=5)    │      │  (5 Regimes)     │             │
│  └──────────────────┘      └──────────────────┘             │
│           │                                                  │
│           ├─────────────┬──────────────┬─────────────┐      │
│           ▼             ▼              ▼             ▼      │
│        Regime 0      Regime 1      Regime 2      Regime n  │
│        (State 0)     (State 1)     (State 2)     (State n) │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
                ▼                            ▼
    ┌────────────────────────┐   ┌──────────────────────┐
    │compute_transition_     │   │fit_state_            │
    │matrix()                │   │distributions()       │
    │                        │   │                      │
    │ Transition Matrix      │   │ Gaussian Models      │
    │ (Regime Probabilities) │   │ (Mean, Std per State)│
    └────────────────────────┘   └──────────────────────┘
                │                            │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   simulate_market()        │
                │  - Sample states           │
                │  - Sample returns          │
                │  - Generate price paths    │
                └────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Synthetic Paths │
                    │   (Monte Carlo)  │
                    └──────────────────┘
```

### Pipeline Architecture

**Training / Analysis Phase:**
```
Raw Market Data → Feature Extraction → Clustering → State Models
       ↓              ↓                   ↓            ↓
     OHLC          Momentum +        Regime          Transitions
                   Volatility        Analysis        + Distributions
```

**Simulation / Forecasting Phase:**
```
Learned Models → State Sampling → Return Sampling → Price Paths
      ↓              ↓              ↓                  ↓
Transitions +    Transition    Gaussian         Monte Carlo
Distributions    Matrix        Distribution     Paths
```

### Component Breakdown

| Component | Function | Input | Output |
|-----------|----------|-------|--------|
| **Data Ingestion** | Fetch NIFTY 50 OHLC | QuestDB | DataFrame |
| **Feature Engine** | Compute momentum & volatility | Prices | Feature Matrix |
| **Clustering** | K-means regime identification | Features | Regime Labels |
| **Transition Model** | Learn regime switching probabilities | Labels | Transition Matrix |
| **Distribution Fitting** | Model returns per regime | Returns + Labels | Gaussian Parameters |
| **Simulator** | Generate synthetic paths | Models + n_steps | Synthetic Returns |

### Data Flow

1. **Input**: OHLC price series from QuestDB
2. **Processing**: Feature transformation → Standardization → Clustering
3. **Analysis**: Transition matrix computation → Distribution fitting
4. **Output**: Market regime model ready for simulation/forecasting

## Project Structure

```
regime_finder/
├── main.py              # Core implementation
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## License

This project is currently unlicensed. See individual files for details.

## Notes

- The database connection defaults assume a local QuestDB instance. Adjust as needed for your setup.
- Multiple implementation approaches for data fetching are included (commented out) for reference
- Default number of regimes is 5; adjust `n_states` parameter based on your analysis needs
