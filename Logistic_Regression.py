import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Aux_Function.PCAM import *

def Logistic_Regression(X,y,stop,initial_guess = None,max_iter=1000, error_diff=1e-3,step=0.001):

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
    Calculate the accuracy of predictions.

    Parameters:
    y_true : np.array
        Array of true labels (ground truth).
    y_pred : np.array
        Array of predicted labels.

    Returns:
    float
        Accuracy as a percentage.
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

# for col in df.columns:
#     if col != 'diagnosis':
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='diagnosis', y=col, data=df)
#         plt.title(f'{col} by Diagnosis')
#         plt.show()

#Changing the categorical values

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


guess1 = (np.random.rand(len(all_features)+1) * 0.01).reshape(-1, 1)

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

# Calculate accuracy
accuracy_sk_pc = accuracy_score(y_test_pc, predictions_sk_pc)
print(f'Accuracy: sklearn {accuracy_sk_pc}')