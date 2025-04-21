import pandas as pd
import matplotlib.pyplot as plt
import pickle
from prepare_feature_matrix import prepare_feature_matrix
from env import AssetAllocationEnv
from ppo import PPOAgent
import traceback
def main():
    # Step 1: Load investment horizon and risk tolerance
    input_df = pd.read_csv("inputs/risk_input.csv")
    investment_horizon = input_df["Investment Horizon"].iloc[0]
    credit_score = input_df["Credit Score"].iloc[0]
    # Convert credit score to risk category
    if credit_score >= 750:
        risk = "Low"
    elif credit_score >= 650:
        risk = "Moderate"
    else:
        risk = "High"
    # Step 2: Prepare feature matrix and asset data
    feature_vector, asset_data, vol_array = prepare_feature_matrix(investment_horizon, risk)
    print("[DEBUG] Merged Asset Data:")
    print(asset_data.head())
    asset_data.to_csv("merged_asset_data.csv")
    print("[INFO] Saved merged asset matrix to merged_asset_data.csv")
    risk_score = feature_vector["risk_score"]
    horizon_days = feature_vector["horizon_days"]
    # Step 3: Save to pickle (optional)
    with open("feature_data.pkl", "wb") as f:
        pickle.dump((feature_vector, asset_data), f)
    # Step 4: Set up environment and agent
    env = AssetAllocationEnv(asset_data, asset_volatility=vol_array, risk_score=risk_score)
    agent = PPOAgent(env)
    agent.train(episodes=100)
    # Step 5: Predict portfolio allocation
    final_state = env.reset()
    final_alloc = agent.choose_action(final_state)
    # Step 6: Save and visualize results
    try:
        print("[DEBUG] Final allocation:", final_alloc)
        print("[DEBUG] Asset columns:", asset_data.columns.tolist())
        results_df = pd.DataFrame({
            "Asset": asset_data.columns,
            "Allocation (%)": [round(x * 100, 2) for x in final_alloc]
        })
        print("[DEBUG] Results dataframe:")
        print(results_df)
        results_df.to_csv("rl_results.csv", index=False)
        print("[INFO] Saved rl_results.csv")
        plt.figure(figsize=(6, 6))
        plt.pie(results_df["Allocation (%)"], labels=results_df["Asset"], autopct="%1.1f%%")
        plt.title("RL Portfolio Allocation")
        plt.tight_layout()
        plt.savefig("allocation.png")
        print("[INFO] Saved allocation.png")
    except Exception as e:
        print(f"[ERROR] Failed to generate plot or CSV: {e}")
        traceback.print_exc()
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] RL pipeline failed: {e}")
        traceback.print_exc()
