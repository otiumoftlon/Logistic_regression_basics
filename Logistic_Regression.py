import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Aux_Function.PCAM import *

def Logistic_Regression(X,y,stop,initial_guess = None,max_iter=1000, error_diff=1e-3,step=0.001):
    """
    Performs logistic regression using gradient descent to optimize the parameters of the model.
    The function iteratively minimizes the log-loss (cross-entropy) and updates the parameters 
    until the specified stopping criteria are met.

    Parameters:
    -----------
    X : array-like (NumPy array, DataFrame)
        The feature matrix where each row is a data point and each column is a feature.

    y : array-like (NumPy array, Series)
        The target vector containing binary labels (0 or 1) for each data point.

    stop : float
        The target error threshold. The algorithm will stop when the log-loss falls below this value.

    initial_guess : array-like, optional, default=None
        The initial values of the model parameters (weights and bias). If None, the parameters are initialized to zeros.

    max_iter : int, optional, default=1000
        The maximum number of iterations for gradient descent. The algorithm will stop if this limit is reached.

    error_diff : float, optional, default=1e-3
        The minimum difference in loss between consecutive iterations. If the difference between the loss of two consecutive
        iterations is less than this value, the algorithm assumes it has converged and stops.

    step : float, optional, default=0.001
        The learning rate or step size for updating the parameters in each iteration.

    Returns:
    --------
    pa : list of NumPy arrays
        A list containing the parameter values at each iteration of gradient descent.

    losses : list of floats
        A list containing the log-loss values at each iteration of the optimization process.

    Process:
    --------
    1. Converts the input data `X` and `y` into NumPy arrays (if they are not already in that format).
    2. Initializes model parameters as zeros (or uses `initial_guess` if provided).
    3. Adds a bias term (column of ones) to the feature matrix `X`.
    4. Computes predictions using the logistic sigmoid function: `predictions = 1 / (1 + exp(-z))`, where `z = X @ parameters`.
    5. Calculates the log-loss (cross-entropy) and tracks the change in loss to determine convergence.
    6. Performs gradient descent to iteratively adjust parameters: `parameters -= step * gradient`, where `gradient = X.T @ (predictions - y)`.
    7. Plots the error over iterations to visualize the learning progress.
    8. Stops if the loss drops below `stop`, the maximum number of iterations is reached, or the loss difference is below `error_diff`.

    Note:
    -----
    - A small value `epsilon = 1e-15` is used to avoid taking the logarithm of zero, which would cause computational errors.
    - The error (log-loss) is recalculated after each parameter update, and the model parameters are stored at each step for future analysis.
    - The plot of the loss function updates every 20 iterations to show how the error decreases over time.

    Example Usage:
    --------------
    # X and y are your data (features and target)
    pa, losses = Logistic_Regression(X, y, stop=0.1, step=0.01, max_iter=500)
    """

    X = X.to_numpy()  # Convert feature matrix to NumPy array if it's a DataFrame
    y = y.to_numpy()  # Convert target vector to NumPy array if it's a Series
    y = y.reshape(-1, 1)
    m,n = np.shape(X)
    if initial_guess is None:
        parameters = np.zeros((n+1,1))
    else:
        parameters = initial_guess
    
    X = np.hstack((X, np.ones((m,1)))) 
    z=X@parameters
    predictions = 1/(1+np.exp(-z))
    epsilon = 1e-15  # A small value to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -1/m * ( y.T@np.log(predictions) + (1-y).T@np.log(1-predictions))
    pa = [parameters]
    losses = [loss[0][0]]
    iterations = 0
    prev_error = loss[0][0] 
    plt.ion()
    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error Over Iterations')

    while loss[0][0] > stop:
  
        error_vector = predictions - y
        gradient = 1/m*X.T@error_vector
        parameters -= step*gradient
        pa.append(parameters)
        z=X@parameters
        predictions = 1/(1+np.exp(-z))
        
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -1/m * ( y.T@np.log(predictions) + (1-y).T@np.log(1-predictions))
        losses.append(loss[0][0])
        if iterations % 20 == 0:
            plt.plot(losses, color='blue')
            plt.xlim(0, len(losses))  # Adjust x-limits
            plt.ylim(0, max(losses) * 1.1)  # Adjust y-limits
            plt.pause(0.01)  # Pause to update the plot
 
        iterations += 1
        if abs(prev_error - loss) < error_diff:
            print(f"Convergence reached after {iterations} iterations.")
            print(f'Coefficients: {parameters}')
            print(f'Loss: {loss}')
            break

        # Update the previous error for the next iteration
        prev_error = loss[0][0]
        if iterations == max_iter:
            print("Max iterations reached.")
            print(f'Coefficients: {parameters}')
            print(f'Loss: {loss}')
            break    
        if loss[0][0] <= stop:
            print(f"Stop reached after {iterations} iterations.")
            print(f'Coefficients: {parameters}')
            print(f'Loss: {loss}')
            break  

    return pa,losses

def calculate_accuracy(y_true, y_pred,threshold = 0.5):
    """
    Calculate the accuracy of binary classification predictions based on a threshold.

    This function compares the true labels (`y_true`) to the predicted probabilities (`y_pred`).
    The predictions are converted into binary class labels (0 or 1) using the specified threshold, 
    and the accuracy is calculated as the proportion of correct predictions.

    Parameters:
    -----------
    y_true : np.array
        A NumPy array containing the true labels (ground truth), with binary values (0 or 1).
    
    y_pred : np.array
        A NumPy array containing the predicted probabilities or scores (floating point values between 0 and 1).
    
    threshold : float, optional, default=0.5
        The decision threshold for converting predicted probabilities into binary class labels.
        Predictions greater than or equal to this threshold are classified as 1, otherwise as 0.
    
    Returns:
    --------
    float
        The accuracy of the predictions, expressed as a fraction of correct predictions (between 0 and 1).
        To get the accuracy as a percentage, multiply the result by 100.
    
    Example:
    --------
    >>> y_true = np.array([0, 1, 0, 1, 1])
    >>> y_pred = np.array([0.4, 0.7, 0.1, 0.9, 0.3])
    >>> calculate_accuracy(y_true, y_pred, threshold=0.5)
    0.8  # 80% accuracy
    
    Notes:
    ------
    - This function assumes binary classification. For multi-class classification, you would need to modify
      the thresholding logic accordingly.
    - The threshold parameter allows flexibility in adjusting the cutoff point for classification, 
      which can be useful in cases where the default threshold of 0.5 does not yield optimal results.
    """
    # Ensure the arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Count the number of correct predictions
    y_pred_class = (y_pred >= threshold).astype(int)

    correct_predictions =  np.sum([1 if i == j else 0 for i, j in zip(y_true, y_pred_class)])

    # Calculate the accuracy
    accuracy = correct_predictions / len(y_true)

    return accuracy

df = pd.read_csv('Logistic Regression\data.csv',index_col='id')

df.pop('Unnamed: 32')

df['diagnosis'] = df['diagnosis'].replace({'B': 0, 'M': 1})
target = df.pop('diagnosis')

df_stand = (df - df.mean())/df.std()
df_stand['diagnosis'] = target
corr_matrix = df_stand.corr()


corr_features = corr_matrix['diagnosis'].sort_values(ascending=False)
target = df_stand.pop('diagnosis')

possible_features = list(corr_features[1:13].index)
all_features = list(corr_features[1:].index)

X_train, X_test, y_train, y_test = train_test_split(df_stand[all_features],target,test_size=0.3,random_state=42)

log_reg = LogisticRegression(max_iter=1000) 
log_reg.fit(X_train, y_train)
coefficients = log_reg.coef_
predictions_sk = log_reg.predict(X_test)

# Calculate accuracy
accuracy_sk = accuracy_score(y_test, predictions_sk)
print(f'Accuracy: sklearn {accuracy_sk}')

guess = np.array([[0.8],  [0.5],   [1.1],  [0.80],  [0.3],  [0.7],
   [0.36060336],  [0.42428622],  [0.7],   [0.9],  [-0.69],  [-0.08],
   [1.21],  [0.565],  [0.88],   [1.3],  [0.59],  [1.194],
   [0.37],  [0.46],  [0.17], [-0.2],  [0.2], [-0.7],
   [0.10], [-0.70], [-0.51], [-0.1], [-0.1],  [0.16],[-0.3]])



guess2 = np.array([[ 0.40392482],[ 0.41266399],[ 0.39714751],[ 0.43587837],[ 0.36510169],[ 0.40506256],[ 0.36778254],
 [ 0.36331652],[ 0.29575821],[ 0.31030392],[ 0.11823894],[ 0.21039784],[ 0.34552401],[ 0.27873287],[ 0.30750716],[ 0.42831375],[ 0.3026845 ],[ 0.31152481],[ 0.337624  ],[ 0.06620303],
 [ 0.16083588],[ 0.08504965],[ 0.08159758],[-0.09701599],[-0.06611082],[-0.18872812],[-0.06416024],[ 0.0174786 ],[-0.16918578],[-0.00568988],[-0.31386277]])
coef, acc = Logistic_Regression(X_train,y_train,stop=1e-10,max_iter=5000,error_diff=1e-6,step=1e-3,initial_guess=guess)

z=X_test@coef[-1][0:-1] + coef[-1][-1]
predictions_me = 1/(1+np.exp(-z))

print('Acuuracy mine',calculate_accuracy(y_test,predictions_me))

plt.ioff() 

plt.figure()
ax = sns.heatmap(corr_matrix, annot=False, cmap='coolwarm',
            center=0,  # Center the colormap at 0 for better visualization
            ) # Make cells square-shaped)

plt.show()

mean = df - np.mean(df , axis=0)

cov_matrix = (mean.T@mean)/(np.shape(mean)[0])
corr_matrix_2 = df.corr()
tab_1,data_pca,f_1 = pca(df,corr_matrix_2,df.columns)
tab_2,data_fa,f_2 = fa(df,corr_matrix_2,df.columns)
print('tabla:','\n',tab_1)
print(f_1)

sns.scatterplot(x=data_pca['PC_1'], y=data_pca['PC_2'],hue=target,alpha=0.4)
plt.show()

X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(data_pca.iloc[:,0:7],target,test_size=0.3,random_state=42)

log_reg_c = LogisticRegression(max_iter=2000) 
log_reg_c.fit(X_train_pc, y_train_pc)

predictions_sk_pc = log_reg_c.predict(X_test_pc)

accuracy_sk_pc = accuracy_score(y_test_pc, predictions_sk_pc)
print(f'Accuracy: sklearn {accuracy_sk_pc}')