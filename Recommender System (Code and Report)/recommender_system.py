'''
Final Project - CS451 
Kailash Raj Pandey, Crystal Paudyal, Stefan Himmel 

Description: 
Using the Netflix Prize Dataset to create a Recommender Systems algorithm 
with vecotorized implementation using numpy and pandas. Also, used the 
fmin_cg algorithm to compare our performance
'''

import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style("darkgrid")


#Specifying Hyper Parameters 
user = 0
numFeatures = 1000
lambd = 1.5
alpha = 0.0025
epoch = 50
probe_filename = "probe.txt"
maxIterFmin = 50
    

def loadData():
    '''
    load the dataset into a pandas DataFrame with 2 colums "Cust_Id" and "Rating" and return it
    '''
    
    #loading 'big220.txt' that only contains ratings for the first 220 movies,
    #using larger dataset would cause MemoryError, even in Gattaca
    
    print("loading big220.txt")
    df1 = pd.read_csv('big220.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    
    #print("loading combined_data_1")
    #df1 = pd.read_csv('combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    #print("loading combined_data_2")
    #df2 = pd.read_csv('combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    #print("loading combined_data_3")
    #df3 = pd.read_csv('combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    #print("loading combined_data_4")
    #df4 = pd.read_csv('combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

    df1['Rating'] = df1['Rating'].astype(float)
    #df2['Rating'] = df2['Rating'].astype(float)
    #df3['Rating'] = df3['Rating'].astype(float)
    #df4['Rating'] = df4['Rating'].astype(float)
    
    df = df1
    #df = df1.append(df2)
    #df = df.append(df3)
    #df = df.append(df4)

    df.index = np.arange(0,len(df))
    print('Full dataset shape: {}'.format(df.shape))
    return df


def cleanData(df):
    '''
    add a new column "Movie_Id" to the DataFrame that associates each rating to a movie
    '''
    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
        temp = np.full((1,i-j-1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Account for last record and corresponding length
    last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
    movie_np = np.append(movie_np, last_record)

    # remove those Movie ID rows
    df = df[pd.notnull(df['Rating'])]

    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)
    
    print("finished cleaning data")
    return df


def spliceData(df):
    '''
    reduces the number of datapoints by removing movies with too less reviews and 
    users who give too less reviews
    '''
    f = ['count','mean']

    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    print('Movie minimum times of review: {}'.format(movie_benchmark))

    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    print('Customer minimum times of review: {}'.format(cust_benchmark))
    
    print('Original Shape: {}'.format(df.shape))
    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]
    print('After Trim Shape: {}'.format(df.shape))
    
    return df


def pivotData(df):
    '''
    convert the DataFrame with just 3 columns into a large matrix with all unique movies on the rows 
    and unique users on the column and the associated ratings in the matrix cell;
    also return another boolean DataFrame, R, of same size as df that contains, in ith row and jth column,
    information about whether ith movie was rated by jth user
    
    '''
    f = ['count','mean']

    print("pivoting the matrix")
    Y = pd.pivot_table(df,values='Rating',index='Movie_Id',columns='Cust_Id')
    print("finished pivoting the matrix")
    
    R = (Y > 0)*1
    print("R shape:",R.shape)
    print("Y shape:",Y.shape)
    return Y, R


def readMovies(filename):
    '''
    read the movie titles from the provided file and puts them in a DataFrame
    '''
    df_title = pd.read_csv('movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
    df_title.set_index('Movie_Id', inplace = True)
    df_title = df_title.as_matrix()
    return df_title


def randInit(numCustomers, numMovies, numFeatures):
    '''
    randomly initialize parameters X and Theta 
    '''
    X = pd.DataFrame(np.random.uniform(-0.1,0.1,size=(numMovies, numFeatures)))
    print("X shape:", X.shape)
    Theta = pd.DataFrame(np.random.uniform(-0.1,0.1,size=(numCustomers, numFeatures)))
    print("Theta shape", Theta.shape)
    return (Theta,X)


def normalize(Y):
    '''
    mean normalize Y
    '''
    movies, cust = Y.shape
    mu = Y.mean(axis = 1)
    Y = Y.sub(mu, axis=0)
    return Y, mu


def fminCostFunc(params, Y, R, numCustomers, numMovies, numFeatures, lamd):
    '''
    compute the cost as sum of all the squared errors to be used in fmin_cg
    '''
    X, Theta = reshapeParams(params, numMovies, numCustomers, numFeatures)
    
    transTheta = Theta.transpose()    
    yHat = X.dot(transTheta)
    
    yHat = np.multiply(yHat,R)
    
    cost = 0.5*(np.nansum(np.square(yHat-Y)))
    cost += (lambd/2.0)*(np.nansum(np.square(Theta)))
    cost += (lambd/2.0)*(np.nansum(np.square(X)))
    
    return cost


def fminGradient(params, Y, R, numCustomers, numMovies, numFeatures, lamd):
    '''
    compute gradients and then flatten the parameters to use with fmin_cg 
    '''
    X, Theta = reshapeParams(params, numMovies, numCustomers, numFeatures)
    transTheta = Theta.transpose()    
    yHat = X.dot(transTheta)
    yHat = np.multiply(yHat,R)
    yHat -= Y
    X_grad = yHat.dot(Theta) + lambd*X
    Theta_grad = yHat.transpose().dot(X) + lambd*Theta

    return flattenParams(X_grad, Theta_grad)


def ourCostFunc(Y,Theta,X,lambd,R):
    '''
    computes cost and the gradients to be uesd in our gradient descent algorithm   
    '''
    
    transTheta = Theta.transpose()    
    yHat = X.dot(transTheta)

    
    yHat = np.multiply(yHat,R)
    
    cost = 0.5*(np.sum(np.square(yHat-Y)))
    cost += (lambd/2.0)*(np.sum(np.square(Theta)))
    cost += (lambd/2.0)*(np.sum(np.square(X)))
    
    yHat -= Y
    
    X_grad = yHat.dot(Theta) + lambd*X
    Theta_grad = yHat.transpose().dot(X) + lambd*Theta
    
    return cost, X_grad, Theta_grad
  

def ourGradDesc(Y, Theta, X, lambd,alpha, epoch, R):
    '''
    compute Gradient Descent to find minimum cost, 
    return the final X and Theta to be used in our algorithm
    '''
    costList = []
    rmseList = []
    
    for i in range(epoch):
        cost, X_grad, Theta_grad = ourCostFunc(Y, Theta, X, lambd, R)
        ourRMSE = rmse(Y, Theta, X, R)
        X -= alpha*(X_grad)
        Theta -= alpha*(Theta_grad)
        print()
        print("epoch",i+1)
        print("cost:", cost)
        print("RMSE:", ourRMSE)
        costList.append(cost)
        rmseList.append(ourRMSE)
        
    plotGraph(costList, rmseList)
    return X, Theta


def rmse(Y, Theta, X, R):    
    '''
    calculate the Root Mean Squared Error 
    '''
  
    transTheta = Theta.transpose()
    yHat = X.dot(transTheta) 
    yHat = np.multiply(yHat,R)
    yHat -= Y
    squaredError = np.square(yHat)
    mse = np.sum(squaredError)/np.sum(R)
    rmse = np.sqrt(mse)
    return rmse


def plotGraph(costList, rmseList):
    '''
    plot the cost and rmse against multiple epoches
    '''
    matplotlib.pyplot.plot(range(len(costList)),costList, "r-")    
    matplotlib.pyplot.xlabel("Iterations")  
    matplotlib.pyplot.ylabel("Cost (J)") 
    matplotlib.pyplot.savefig("cost.png")
    matplotlib.pyplot.close()
    
    matplotlib.pyplot.plot(range(len(rmseList)), rmseList, "b-")
    matplotlib.pyplot.xlabel("Iterations")  
    matplotlib.pyplot.ylabel("RMSE")
    matplotlib.pyplot.savefig("rmse.png")
    matplotlib.pyplot.close()


def flattenParams(X, Theta):
    '''
    flatten X and Theta to be used in fmincg
    '''
    return np.concatenate((X.flatten(),Theta.flatten()))


def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
    '''
    unroll parameters to retrieve X and Theta
    '''
    reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))
    reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))
    
    return reX, reTheta


def recommend(user, prediction_matrix, mu, movies,R):
    '''
    recommend top 10 movies for the user
    '''
    user_predictions = prediction_matrix[:,user] + mu.flatten()
    userR = R[:,user]
    
    #inverting R to only consider non-rated movies for recommendation
    userR = -1*userR + 1
    
    user_predictions = np.multiply(user_predictions, userR)
    # Sort user predictions from highest to lowest
    pred_idxs_sorted = np.argsort(user_predictions)
    pred_idxs_sorted[:] = pred_idxs_sorted[::-1]

    print("Top recommendations for you:")
    for i in range(10):
        print('Predicting rating %0.1f for movie %s.' % 
        (user_predictions[pred_idxs_sorted[i]],(movies[pred_idxs_sorted[i]][1])))


def probe(probe_filename, finalYHat, movieIds, userIds, Y):
    '''
    calculate RMSE for the probe dataset
    '''
    se = 0
    count = 0
    userDict = dict(zip(userIds,range(len(userIds))))
    movieDict = dict(zip(movieIds, range(len(movieIds))))
    skipMovie = False
    with open(probe_filename, "r") as file:
        file = file.readlines()
        for line in file:
            line = line.strip()
            if ":" in line:
                skipMovie = False
                currMovie = int(line[:-1])
                if currMovie not in movieDict:
                    skipMovie = True
                else:
                    row = movieDict[currMovie]
                    #print("movie:", currMovie)
            else:
                if (not skipMovie):
                    currUser = int(line)
                    if currUser in userDict:
                        col = userDict[currUser]
                        se += (Y[row, col] - finalYHat[row, col])**2
                        count += 1
        mse = se/count
        rmse = np.sqrt(mse)
    return rmse
        

def ourMain():
    '''
    main function to drive our algorithm
    '''
    
    print("loading dataset...")
    df = loadData()
    print("done")
    
    print("cleaning dataset...")
    cleanDf = cleanData(df)
    print("done")
    
    print("splicing dataset...")
    splicedDf = spliceData(cleanDf)
    print("done")
    
    print("pivoting dataset...")
    Y, R = pivotData(splicedDf)
    print("done")
    
    print("reading movies...")
    movies = readMovies("movie_titles.csv")
    numMovies, numCustomers = Y.shape
    print("done")
    
    print("randomly initializing Theta and X...")
    Theta,X = randInit(numCustomers, numMovies, numFeatures)
    print("done")
    
    print("normalizing Y...")
    Y, mu = normalize(Y)
    print("done")
    
    movieIds = Y.index
    userIds = Y.columns.values
    Y = Y.as_matrix()
    Theta = Theta.as_matrix()
    R = R.as_matrix()
    X = X.as_matrix()
    mu = mu.as_matrix()
    
    Y[np.isnan(Y)] = 0    
    
    print("running gradient descent...")
    X, Theta = ourGradDesc(Y, Theta, X, lambd,alpha,epoch,R)
    print("done")
    
    finalYHat = X.dot(Theta.transpose())
    
    print("recommending user...")
    recommend(user, finalYHat, mu, movies,R)
    print("done")
    
    print("calculating RMSE on the probe dataset...")
    probeRMSE = probe(probe_filename, finalYHat, movieIds, userIds, Y)
    print("done")
    print("RMSE for probe dataset:",probeRMSE)
    
    
def fminMain():
    '''
    main function to drive fmin_cg algorithm
    '''
    print("loading dataset...")
    df = loadData()
    print("done")
    
    print("cleaning dataset...")
    cleanDf = cleanData(df)
    print("done")
    
    print("splicing dataset...")
    splicedDf = spliceData(cleanDf)
    print("done")
    
    print("pivoting dataset...")
    Y, R = pivotData(splicedDf)
    print("done")
    
    print("reading movies...")
    movies = readMovies("movie_titles.csv")
    numMovies, numCustomers = Y.shape
    print("done")
    
    print("randomly initializing Theta and X...")
    Theta,X = randInit(numCustomers, numMovies, numFeatures)
    print("done")
    
    print("normalizing Y...")
    Y, mu = normalize(Y)
    print("done")
    
    movieIds = Y.index
    userIds = Y.columns.values
    Y = Y.as_matrix()
    Theta = Theta.as_matrix()
    R = R.as_matrix()
    X = X.as_matrix()
    mu = mu.as_matrix()
    
    Y[np.isnan(Y)] = 0    
        
    flatParams = flattenParams(X,Theta)
    
    print("running advanced optimization algorithm...")
    result = scipy.optimize.fmin_cg(fminCostFunc, x0=flatParams, fprime=fminGradient, 
                               args=(Y,R,numCustomers,numMovies,numFeatures,lambd), 
                                maxiter=maxIterFmin,disp=True,full_output=True)
    print("done")
    
    print(result)
    X, Theta = reshapeParams(result[0], numMovies, numCustomers, numFeatures)
    RMSE = rmse(Y, Theta, X, R)
    print("RMSE:",RMSE)


if __name__ == "__main__":
    #ourMain()
    fminMain()
