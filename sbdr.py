import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import json

def computeCov (X):
    """
    Computes the variance-covariance matrix. (or use numpy.cov)
    Arguments:
      X -- Set of M N-dimensional data vectors. Shape: (M,N)
    Returns:
      Cx -- Covariance matrix. Shape: (N,N)
    """
    X_t = np.transpose(X - np.sum(X, axis=0)/X.shape[0])
    Cx = np.dot(X_t, X)/(X.shape[0])
    return Cx

def computeCovBatch (X):
    """
    Batch computation of the variance-covariance matrix
    Arguments:
      X -- Batch of Np data matrices. Shape: (Np,M,N)
    Returns:
      Cx -- Batch of Np covariance matrices. Shape: (Np,N,N)
    """
    Np,M,N = X.shape
    Cx = np.ones((Np,N,N))
    for i in range (Np):
        Cx[i,:,:] = computeCov(X[i,:,:])
    return Cx

def computeCovBatchGivenPX (P, X):
    """
    Batch computation of the new data set Y and its covariance matrix Cy, given the transformation matrix P and the data set X.
    Arguments:
      P -- Batch of Np transformation matrices. Shape (Np,N,K)
      X -- Batch of Np data matrices. Shape (Np,M,N)
    Returns:
      Y -- Batch of Np data matrices. Shape (Np,K,K)
      Cy -- Batch of Np covariance matrices. Shape (Np,K,K)
    """
    Np,M,N = X.shape
    K = P.shape[2]
    Y = np.zeros((Np,M,K))
    for j in range(Np):
        Y[j,:,:] = np.dot(X[j,:,:],P[j,:,:])
    Cy = np.ones((Np,K,K))
    for i in range (Np):
        Cy[i,:,:] = computeCov(Y[i,:,:])
    return Y, Cy

def fitness(Cy, alpha1=5, alpha2=1, theta1=1, theta2=1):
    """
    Evaluates the fitness of the solution. By default, it makes sure we are maximizing the variance and minimizing the covariance.
    Arguments:
      Cy -- Covariance matrix. Shape: (N,N)
            Note - Shape could also be expressed as (K,K), depending on whether it is evaluating Cx or Cy, as X is N-dimensional while Y is K-dimensional.
	  alpha1 -- variable that controls minimization/maximization of the variance.
      alpha2 -- variable that controls minimization/maximization of the covariance.
      theta1 -- scaling coefficient that controls the contribution of the main diagonal elements in the final score
      theta2 -- scaling coefficient that controls the contribution of the off-diagonal elements in the final score
	Returns:
	  Fitness score
    """
    # 1st approach - SNR
    # return SNR(Cy)
    
    # 2nd approach - weighted summation:
    A = np.sum(np.diagonal(np.abs(Cy)))
    B = np.sum(np.abs(Cy)) - A
    return math.log(alpha1*A - alpha2*B)

def SNR(Cy):
    A = np.sum(np.diagonal(np.abs(Cy)))
    B = np.sum(np.abs(Cy)) - A
    return math.log(A/B)

def fitnessBatch(Cy, alpha1=5, alpha2=-5, theta1=1, theta2=1):
    """
    Evaluates the fitness of a batch of Np particles.
    Arguments:
      Cy -- Batch of Np covariance matrices. Shape (Np,K,K).
    Returns:
      F -- fitness scores for the Np particles, shape(Np,1)
    """
    Np,K,K = Cy.shape
    F = np.ones((Np,1))
    for i in range (Np):
        F[i] = fitness(Cy[i,:,:], alpha1, alpha2, theta1, theta2)
    return F

def invisible_walls(P, boundaries):
    """
    Implementation of the invisible walls boundary checking
    Arguments:
      P -- batch of Np particles. Shape: (Np,N,K)
      boundaries -- boundary conditions. Shape: (N,K,2)
    Returns:
      P -- batch of Np particles. Shape: (Np,N,K)
    """
    for j in range(P.shape[1]):
        particles_crossed = np.nonzero(np.logical_or(P < boundaries[j,0], P > boundaries[j,1]))[0]
        P[particles_crossed] = np.random.rand(particles_crossed.shape[0],P.shape[1])*(boundaries[j,1]-boundaries[j,0])+boundaries[j,0]
    return P

def initializePSO(Np, M, N, X):
    """
    Initializes the parameters
    Arguments:
      Np -- number of particles
      M -- dimensionality of the particle
      X -- original data set. Shape: (M,N)
    Returns:
      P -- batch of Np particles, shape (Np,N,K)
      X -- Standardized data set. Shape: (M,N)
      X_batch -- Batch of Np X matrices. Shape: (Np,M,N)
      pbest_particles -- initial personal bests, shape (Np,N,K)
      pbest_fitness -- initial personal best scores, shape (Np,1)
      gbest_particle -- initial global best, shape (1,N,K)
      gbest_fitness -- initial global best score, integer
      boundaries -- boundary conditions for each of the Np particles along each of the K dimensions. Shape: (Np,K,2)
      V -- Velocity of each of the Np particles. Shape: (Np,N,K)
    """
    # Standardize X
    X = StandardScaler().fit_transform(X)

    # Cascade a batch of Np X
    X_batch = np.tile(X,(Np,1,1)) # replicate X Np times along the 1st direction, with only 1 replication along the 2nd and 3rd dimensions

    # Generate batch of Np particles (transformation matrices) P
    boundaries = 1*np.ones((N,K,2)) # for each of the N and K dimensions, a lower and an upper boundary
    boundaries[:,:,0] = -1*boundaries[:,:,0]

    P = np.random.rand(Np,N,K)*(boundaries[0,0,1]-boundaries[0,0,0])+boundaries[0,0,0] #generateTransMatUsingStdPCA(X, K)

    # Compute batch of covariance matrices Cy
    Y, Cy = computeCovBatchGivenPX (P, X_batch)

    # Initialize pbest and gbest
    pbest_particles = P
    pbest_fitness = fitnessBatch(Cy)
    gbest_fitness = pbest_fitness[np.argmax(pbest_fitness, axis=0)]
    gbest_particle = pbest_particles[np.argmax(pbest_fitness, axis=0)]

    V = np.zeros((Np,N,K))

    return P, X, X_batch, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V

def runPSO(P, X_batch, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V, I, M, N, w, c1, c2, dt, boundary_condition):
    """
    Runs PSO for a number of epochs
    Arguments:
      P -- batch of Np particles. Shape: (Np,N,K)
      pbest_particles -- Initial personal bests. Shape: (Np,N,K)
      pbest_fitness -- Initial personal best scores, shape (Np,1)
      gbest_particle -- Initial global best, shape (1,N,K)
      gbest_fitness -- Initial global best score, integer
      I -- Number of epochs
      M -- Number of particles
      N -- Dimensionality of the particle
    Returns:
      gbest_particle -- global best particle. Shape: (N,K)
      gbest_fitness -- fitness score achieved by the global best particle
    """

    for i in range(I):
        Y, Cy = computeCovBatchGivenPX (P, X_batch)
        # Evaluate current particles' fitness
        current_fitness = fitnessBatch(Cy)

        # Find the new personal best fitness and deduce the new global best fitness
        pbest_fitness_prev = pbest_fitness
        pbest_fitness = np.maximum(current_fitness, pbest_fitness_prev)
        gbest_fitness = pbest_fitness[np.argmax(pbest_fitness, axis=0)]

        # Update personal best particles and deduce the new global best particle
        outperforming_particles = np.greater(pbest_fitness, pbest_fitness_prev).reshape(Np,)
        pbest_particles[outperforming_particles, :, :] = P[outperforming_particles, :, :]
        gbest_particle = pbest_particles[np.argmax(pbest_fitness, axis=0),:, :]

        # Update the velocities of the particles
        V = w*V + c1*np.random.rand(Np,N,K)*(pbest_particles - P) + c2*np.random.rand(Np,N,K)*(gbest_particle - P)

        # Move the particles
        P = P + dt*V

    return gbest_particle, gbest_fitness

def runPCA (X, Y, K):
    """
    Runs PCA using sklearn.
    """
    print('\nPCA:')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA(n_components=K)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    printTrainingPrediction (X_test, Y_test, X_train, Y_train)
    
    X_gbest = np.concatenate((X_train, X_test))
    Cx_gbest = computeCov(X_gbest)
    f = fitness(Cx_gbest)
    print('Fitness: ', "%.2f" % f)

def runSBDR (X, Y, K):
    """
    Runs SBDR.
    """
    M, N = X.shape

    P, X, X_batch, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V = initializePSO(Np, M, N, X)

    gbest_particle, gbest_fitness = runPSO(P, X_batch, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V, I, M, N, w, c1, c2, dt, boundary_condition)

    P_gbest = gbest_particle[0,:,:]
    Y_gbest = np.dot(X, P_gbest)

    X_train, X_test, Y_train, Y_test = train_test_split(Y_gbest, Y, test_size=0.2, random_state=0)

    Cy_gbest = computeCov(Y_gbest)
    f = fitness(Cy_gbest)
    
    acc = getAccuracyOfTrainingPredictions(X_test, Y_test, X_train, Y_train)
    return f, acc

def printTrainingPrediction (X_test, Y_test, X_train, Y_train):
    """
    Trains a random forest model and makes predictions on the test set.
    """
    # Initialize classifier
    classifier = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=1000)
    # Training:
    classifier.fit(X_train, Y_train)
    # Predicting over the test set
    Y_pred = classifier.predict(X_test)
    # Print confusion matrix and accuracy score:
    print('Accuracy: ', accuracy_score(Y_test, Y_pred))




def getAccuracyOfTrainingPredictions (X_test, Y_test, X_train, Y_train):
    """
    Trains a random forest model and makes predictions on the test set
    and saves the accuracy score to CSV file.
    """
    # Initialize classifier
    classifier = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=1000)
    # Training:
    classifier.fit(X_train, Y_train)
    # Predicting over the test set
    Y_pred = classifier.predict(X_test)
    # Print confusion matrix and accuracy score:
    acc = accuracy_score(Y_test, Y_pred)
    
    return acc

if __name__ == "__main__":
    # Load parameters
    with open('config.json') as json_file:
        data = json.load(json_file)
        K = data["K"]
        Np = data["Np"]
        I = data["I"]
        w = data["w"]
        c1 = data["c1"]
        c2 = data["c2"]
        dt = data["dt"]
        boundary_condition = data["boundary_condition"]
        dataset_path = data["dataset_path"]

    # Load data set
    dataset = pd.read_csv(filepath_or_buffer=dataset_path, header=None, sep=',')
    X = dataset.drop(columns=4)
    Y = dataset[4]
    
    # Run PCA
    runPCA(X,Y,K)
    
    # RUN SBDR
    print('\nSBDR:')
    df = pd.DataFrame(columns=['Fitness','Accuracy'])
    
    R = 1000
    for i in range(R):
        if (i==0.25*R): print('25% completed')
        elif (i==0.5*R): print('50% completed')
        elif (i==0.75*R): print('75% completed')
        elif (i==0.95*R): print('95% completed')
        f, acc = runSBDR(X,Y,K)
        df_curr = pd.DataFrame([[f, acc]], columns=['Fitness','Accuracy'])
        df = pd.concat([df, df_curr], ignore_index=True) # Deprecated: df = df.append(df_curr, ignore_index=True)
    df.to_csv("test.csv")
    
    fitness_avg = df.sum(axis=0)/R
    print('\nSBDR Avg. Accuracy Score: ', fitness_avg['Accuracy'])
    print('SBDR Avg. Fitness Score: ', fitness_avg['Fitness'])