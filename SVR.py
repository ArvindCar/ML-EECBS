import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse
import time
from sklearn.metrics import r2_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--map_name', type=str, default='',
                        help='The name map (Not the instance)')
    parser.add_argument('--no_save', action='store_true', default=False,
                        help='The model weights are not stored')
    args = parser.parse_args()
    # Step 1: Load the data from the CSV file
    data = pd.read_csv(f'data_collection_{args.map_name}.csv')

    # Step 2: Preprocess the data if necessary

    # Step 3: Split the data into input features (X) and target variable (y)
    X = data.drop(columns=['cost-to-go', 'f_hat', 'id', 'parent'])  # Input features
    y = data['cost-to-go']  # Target variable
    feature_names = X.columns.tolist()

    with open('feature_names.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(feature_names)

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    h_hat_test = np.array(X_test['h_hat'])
    h_hat_train = np.array(X_train['h_hat'])

    nan_indices_train = np.where(np.isnan(X_train).any(axis=1))[0]
    print("Number of Del Rows: ",len(nan_indices_train))

    # Print the samples with missing values
    for idx in nan_indices_train:
        print(feature_names)
        print("Sample index:", idx)
        print("Features:", X_train.iloc[idx])

    nan_indices_test = np.where(np.isnan(X_test).any(axis=1))[0]
    print("Number of Del Rows: ",len(nan_indices_test))
 

    # Step 5: Define and train the SVR model using the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Scale the features
    X_test_scaled = scaler.transform(X_test)

    nan_indices_train = np.where(np.isnan(X_train_scaled).any(axis=1))[0]
    print("Number of Del Rows: ",len(nan_indices_train))

    # Print the samples with missing values
    for idx in nan_indices_train:
        print(feature_names)
        print("Sample index:", idx)
        print("Features:", X_train_scaled[idx])
        
    X_train_scaled = np.delete(X_train_scaled, nan_indices_train, axis=0)
    y_train = np.delete(y_train, nan_indices_train, axis=0)
    h_hat_train = np.delete(h_hat_train, nan_indices_train, axis=0)

    nan_indices_test = np.where(np.isnan(X_test_scaled).any(axis=1))[0]
    print("Number of Del Rows: ",len(nan_indices_test))
    X_test_scaled = np.delete(X_test_scaled, nan_indices_test, axis=0)
    y_test = np.delete(y_test, nan_indices_test, axis=0)
    h_hat_test = np.delete(h_hat_test, nan_indices_test, axis=0)

    svr = SVR(kernel='linear')  # You can specify other parameters as well
    svr.fit(X_train_scaled, y_train)

    if not args.no_save:
        joblib.dump(svr, 'svm_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

    # Step 6: Evaluate the trained model on the testing data
    y_pred = svr.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)

    plt.scatter(y_test, y_pred, color='blue', alpha=0.8, label = "ML-EECBS Test data")
    plt.scatter(y_train, svr.predict(X_train_scaled), color='cyan', alpha=0.1, label = "ML-EECBS Train data")
    plt.scatter(y_test, h_hat_test, color='green', alpha=0.1, label = "EECBS formula")
    
    # plt.scatter(y_train, h_hat_train, color='green', alpha=0.3)
    plt.plot([0, y_pred.max()], [0, y_pred.max()], color='red', linestyle='--')
    plt.xlim([0, y_pred.max()])
    plt.ylim([0, y_pred.max()])
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    # plt.title(f'Actual vs. Predicted Values ({args.map_name})')
    plt.title('SVR with linear kernel')
    plt.legend()
    plt.text(0.5, 0.90, f'R2 Score: {r2:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    # plt.show()
    plt.savefig(f'SVR_Results_Linear.png')  # Change 'your_plot_filename.png' to your desired filename and format

    # Close the plot to free up memory
    plt.close()
