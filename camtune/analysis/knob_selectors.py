import shap
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score




def GBMSelect(X, y, test_size: float=0.2, n_estimator: int=100, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = GradientBoostingRegressor(random_state=random_state, n_estimators=n_estimator)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    return model


def RFSelect(X, y, n_estimators=100, test_size=0.2, random_state=42):
    """
    Apply Random Forest to dataset to determine feature importance.
    
    Args:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target vector.
    - n_estimators (int): Number of trees in the forest.
    - test_size (float): Fraction of the dataset to hold as test set.
    - random_state (int): Seed for the random number generator.
    
    Returns:
    - random_forest (RandomForestRegressor or RandomForestClassifier): Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    random_forest = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    random_forest.fit(X_train, y_train)
    
    print("Training score:", random_forest.score(X_train, y_train))
    print("Test score:", random_forest.score(X_test, y_test))

    return random_forest

# Rewrite RFSelect to use Cross Validation
def RFSelectCV(X, y, n_estimators=100, cv_folds=5, random_state=42):
    random_forest = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(random_forest, X, y, cv=cv_folds)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores)}")
    print(f"Standard deviation of CV scores: {np.std(cv_scores)}")
    
    # Train the final model on the entire dataset
    random_forest.fit(X, y)
    
    return random_forest

def LassoSelect(X, y, alpha=0.01, test_size=0.2, random_state=42):
    """
    Apply Lasso regression to dataset to determine feature importance.
    
    Args:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target vector.
    - alpha (float): Regularization strength of Lasso.
    - test_size (float): Fraction of the dataset to hold as test set.
    - random_state (int): Seed for the random number generator.
    
    Returns:
    - lasso (Lasso): Trained Lasso model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    lasso = Lasso(alpha=alpha, random_state=random_state)
    lasso.fit(X_train, y_train)
    
    print("Training score:", lasso.score(X_train, y_train))
    print("Test score:", lasso.score(X_test, y_test))
    
    return lasso


def SHAPSelect(X, y, model_class=GradientBoostingRegressor, random_state=1024, test_size=0.15):
    """
    Compute SHAP values for the given dataset and model.
    
    Args:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target variable.
    - model_class (class, optional): Model class to use for training. Default is GradientBoostingRegressor.
    - random_state (int, optional): Random state for reproducibility.
    - test_size (float, optional): Fraction of data to be used as test set.
    
    Returns:
    - shap_values (numpy.ndarray): Computed SHAP values.
    - feature_names (list): List of feature names.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize and train the model
    model = model_class(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Create a SHAP Tree explainer and compute SHAP values
    explainer = shap.TreeExplainer(model)    
    return explainer