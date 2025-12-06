import pandas as pd
from sklearn.model_selection import train_test_split

# Import preprocessing
from trainer.data_preprocessing import preprocess_dataframe

# Import ONE model for now (we’ll add the others later)
from models.linear_model import LinearModel


def main():

    # 1. Load dataset 
    df = pd.read_excel("data/IL_Dataset.xlsx") # replace with your data

 
    # 2. Preprocess the dataframe
    df = preprocess_dataframe(df)

    
    # 3. Split features and target
    X = df.drop(columns=["Personal Best"])
    y = df["Personal Best"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train the Linear Regression model
    model = LinearModel()
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)

    # 5. Print results
    print("Linear Regression Results:")
    print(f"R²:   {results['r2']:.3f}")
    print(f"MAE:  {results['mae']:.2f} seconds")
    print(f"RMSE: {results['rmse']:.2f} seconds")


if __name__ == "__main__":
    main()
