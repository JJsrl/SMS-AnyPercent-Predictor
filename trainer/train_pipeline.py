import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Import preprocessing
from trainer.data_preprocessing import preprocess_dataframe

# Import ALL models
from models.linear_model import LinearModel
from models.forest_model import ForestModel
from models.tree_model import DecisionTreeModel
from models.bagging_model import BaggingModel


def main():
    # 1. Load dataset 
    df = pd.read_excel("data/IL_Dataset.xlsx")
    
    print(f"Original dataset shape: {df.shape}")
    
    # 2. Preprocess the dataframe
    df = preprocess_dataframe(df)
    
    print(f"After preprocessing: {df.shape}")
    
    # 3. Split features and target
    X = df.drop(columns=["Personal Best"])
    y = df["Personal Best"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Define all models
    models = {
        "Linear Regression": LinearModel(),
        "Random Forest": ForestModel(),
        "Decision Tree": DecisionTreeModel(),
        "Bagging": BaggingModel()
    }
    
    # 5. Train all models and collect results
    results_list = []
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print('='*50)
        
        model.train(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        
        # Add model name to results
        results['model'] = name
        results_list.append(results)
        
        # Print results
        print(f"R²:   {results['r2']:.3f}")
        print(f"MAE:  {results['mae']:.2f} seconds ({results['mae']/60:.2f} minutes)")
        print(f"RMSE: {results['rmse']:.2f} seconds ({results['rmse']/60:.2f} minutes)")
        
        # Save feature importances for Random Forest
        if name == "Random Forest":
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance_df.to_csv("results/feature_importances.csv", index=False)
            print("✅ Saved feature importances to results/feature_importances.csv")
    
    # 6. Save results to CSV
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("results/model_results.csv", index=False)
    
    print(f"\n{'='*50}")
    print("✅ All models trained successfully!")
    print("✅ Results saved to results/model_results.csv")
    print('='*50)


if __name__ == "__main__":
    main()