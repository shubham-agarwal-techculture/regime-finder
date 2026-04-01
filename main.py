# setup
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import psycopg2
import asyncpg
from sqlalchemy import create_engine


def fetch_questdb_data():
    engine = create_engine("postgresql+psycopg2://admin:quest@192.168.1.101:8812/qdb")

    query = """
    SELECT date, open, high, low, close
    FROM nifty_50;
    """

    df = pd.read_sql(query, engine)

    return df


# def fetch_questdb_data():
#     conn = psycopg2.connect(
#         host="192.168.1.101",
#         port=8812,  # QuestDB PG port
#         user="admin",
#         password="quest",
#         database="qdb",
#     )

#     query = """
#     SELECT
#         date,
#         open,
#         high,
#         low,
#         close,
#     FROM nifty_50
#     """

#     df = pd.read_sql(query, conn)
#     conn.close()

#     return df


# async def fetch_questdb_data():
#     conn = await asyncpg.connect(
#         host="192.168.1.101",
#         port=9000,  # QuestDB PG port
#         user="admin",
#         password="quest",
#         database="qdb",
#     )

#     query = """
#     SELECT
#         date,
#         open,
#         high,
#         low,
#         close,
#         volume,
#         oi
#     FROM nifty_50
#     ORDER BY date
#     lIMIT 10;
#     """

#     rows = await conn.fetch(query)
#     await conn.close()

#     df = pd.DataFrame([dict(r) for r in rows])
#     print(df)
#     return df

#     # df = pd.read_sql(query, conn)
#     # conn.close()

#     # return df


def compute_features(df, price_col="close"):
    df = df.copy()

    # returns
    df["ret_1"] = df[price_col].pct_change()

    # momentum (multi-horizon)
    df["mom_5"] = df[price_col].pct_change(5)
    df["mom_20"] = df[price_col].pct_change(20)
    df["mom_60"] = df[price_col].pct_change(60)

    # volatility (risk)
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()

    df = df.dropna()
    return df


def cluster_states(df, n_states=5):
    features = ["mom_5", "mom_20", "mom_60", "vol_5", "vol_20", "vol_60"]

    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_states, random_state=42)
    states = kmeans.fit_predict(X_scaled)

    df["state"] = states

    return df, kmeans, scaler


def compute_transition_matrix(states, n_states):
    matrix = np.zeros((n_states, n_states))

    for i in range(len(states) - 1):
        matrix[states[i], states[i + 1]] += 1

    # normalize rows → probabilities
    matrix = matrix / matrix.sum(axis=1, keepdims=True)

    return matrix


def fit_state_distributions(df):
    state_models = {}

    for state in sorted(df["state"].unique()):
        subset = df[df["state"] == state]["ret_1"]

        mean = subset.mean()
        std = subset.std()

        state_models[state] = (mean, std)

    return state_models


def fit_state_distributions(df):
    state_models = {}

    for state in sorted(df["state"].unique()):
        subset = df[df["state"] == state]["ret_1"]

        mean = subset.mean()
        std = subset.std()

        state_models[state] = (mean, std)

    return state_models


def simulate_market(n_steps, transition_matrix, state_models, start_state=0):
    states = [start_state]
    returns = []

    current_state = start_state

    for _ in range(n_steps):
        # sample next state
        probs = transition_matrix[current_state]
        next_state = np.random.choice(len(probs), p=probs)

        # sample return from Gaussian
        mean, std = state_models[next_state]
        ret = np.random.normal(mean, std)

        states.append(next_state)
        returns.append(ret)

        current_state = next_state

    return np.array(states), np.array(returns)


def build_model(df, n_states=5):
    df = compute_features(df)
    print(f"Features computed: \n {df.head()}")
    df, kmeans, scaler = cluster_states(df, n_states)

    transition_matrix = compute_transition_matrix(df["state"].values, n_states)
    state_models = fit_state_distributions(df)

    return {
        "df": df,
        "kmeans": kmeans,
        "scaler": scaler,
        "transition_matrix": transition_matrix,
        "state_models": state_models,
    }


if __name__ == "__main__":
    # df must have a 'close' column

    df = fetch_questdb_data()

    print(f"Data fetched from QuestDB: \n {df.head()}")

    model = build_model(df, n_states=5)

    states, returns = simulate_market(
        n_steps=100,
        transition_matrix=model["transition_matrix"],
        state_models=model["state_models"],
        start_state=0,
    )

    print(f"Simulated states: {states}")
    print(f"Simulated returns: {returns}")
