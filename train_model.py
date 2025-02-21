import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Create a function to save the trained model and scaler
def save_model_and_scaler():
    # Load dataset
    df = pd.read_csv('data/dataset.csv')
    
    # Split the dataset
    from sklearn.model_selection import train_test_split
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the logistic regression model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    with open('heart_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler