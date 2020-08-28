# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:13:04 2020

@author: Armando
"""


import numpy as np
from numpy.random import randn,rand
from random import random
import pandas as pd
import time
import matplotlib.pyplot as plt
#from filterpy import plot_covariance_ellipse

def create_gaussian_particles(mean, std, N):
    '''
    Create Gaussioan Distributed Particles
    
    PARAMETERS
     - mean: Mean of the distributed particles   
     - std:  Standard deviation of the distributed particles
     - N:    Number of particles
    
    DESCRIPTION
    Create N by 2 array to store x location, y location
    of each particle uniformly distributed.
    
    Returns particle locations
    '''
    
    particles = np.empty((N, 2))    
    particles[:, 0] =  mean[0] + (randn(N) * std[0]) #np.random.normal(0, np.sqrt(10), 1)
    particles[:, 1] =  mean[1] + (randn(N) * std[1])
    return particles


def pdf(z,z1,z2,R):
    '''
    Probability distribution function
    
    Parameters
    ----------
    z : The abservation of the radar.
    z1 : The distance d of partilce.
    z2 : The angle thita of particle.
    R : Standard deviation of the observation noise.
    
    DESCRIPTION
    Gaussian distribution function for d and thita
    
    Returns
    -------
    p : The probability of  a particle to be the corrent one.

    '''
    
    #z1=d and z2=thita
    p=(1/np.sqrt(2*np.pi *R[0]))*np.exp((-1/2)*((((z[0]-z1)/R[0])))) * (1/np.sqrt(2*np.pi *R[1]))*np.exp((-1/2)*((((z[1]-z2)/R[1]))))
    return p        



def neff(weights):
    '''
    Calculate effective N
    
    PARAMETERS
     - weights:    Weights of all particles
    
    DESCRIPTION
    Calculates effective N, an approximation for the number of particles 
    contributing meaningful information determined by their weight. When 
    this number is small, the particle filter should resample the particles
    to redistribute the weights among more particles.
    
    Returns effective N.
    '''
    return 1. / np.sum(np.square(weights))



class ParticleFilter:
    
    def __init__(self,N,Q,R,init_pos):
        '''
        Parameters
        ----------
        N : Number of particles
        Q : Standard deviation  of cotrol input u noise
        R : Standard deviation of the observation noise
        init_pos : Initial position of the robot (x,y)

        Returns
        -------
        None.

        '''
        
        self.N=N
        self.Q=Q
        self.R=R
        self.init_pos=init_pos
        
        # Locations of all particles
        self.particles = create_gaussian_particles(init_pos,[0.1,0.1],N)
        # Weights of the particles
        self.weights=np.ones(N)/N
        
    def predict(self,u):
        '''
        Predict the next location of the particles
    
        PARAMETERS
        - u: X location step, and y location step to predict particles
     
        DESCRIPTION
        Predict particles forward one time step using control input u
        (x step, y step) and noise with standard deviation std.
        '''
        
        self.particles[:, 0] = 0.3*self.particles[:, 0] +np.exp(-(self.particles[:,1])**2)+(u[0]+ (randn(self.N)*self.Q[0]))
        self.particles[:, 1] = 0.3*self.particles[:, 1] +np.exp(-(self.particles[:,0])**2)+(u[1]+ (randn(self.N)*self.Q[1]))
        

    def update(self,z):
        '''
        Update particle weights
        
        PARAMETERS
         - z:  Observation of the robot's movement 
        
        DESCRIPTION
        Calculate the d and thita for each particle. Then, calculate the pdf for a measurement of observation. 
        Multiply weight by pdf. If observation is close to particle d and thita, then the 
        particle is similar to the true state of the model so the pdf is close 
        to one so the weight stays near one. If observation is far from d and thita,
        then the particle is not similar to the true state of the model so the 
        pdf is close to zero so the weight becomes very small.   
        '''
        
        
        d=np.sqrt((self.particles[:,0]**2+self.particles[:,1]**2)) # d
        thita=np.arctan2(self.particles[:,1],self.particles[:,0]) # thita
        
        w=[ pdf(z,d[i],thita[i],self.R) for i in range(0,len(self.particles))]
        
        self.weights *= w
        self.weights += 1.e-5    # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize
        
        
        
    def estimate(self):
        '''
        Estimate state of system
    
        DESCRIPTION
        Estimate the state of the system by calculating the mean and variance
        of the weighted particles.
    
        Returns mean and variance of the particle locations
        '''
        
        pos = self.particles[:,0:2]
        mean = np.average(pos,weights = self.weights,axis =0)
        var  = np.average((pos - mean)**2, weights=self.weights, axis=0)
        
        return mean, var
        
    
    def residual_resample(self):
        '''
        Residual resample of particles
        
        DESCRIPTION
        The normalized weights are multiplied by N, and then the integer 
        value of each weight is used to define how many samples of that particle will be. 
        Next we take the residual: the weights minus the integer part, which leaves the 
        fractional part of the number and then use a simpler sampling scheme such as multinomial

        '''
        
        N = self.N
        indexes = np.zeros(N, 'i')
    
        
        num_copies = (N*np.asarray(self.weights)).astype(int)
        k = 0
        
        for i in range(N):
            for _ in range(num_copies[i]): 
                indexes[k] = i
                k += 1
    
        
        residual = self.weights - num_copies     
        residual /= sum(residual)     
        cum_sum = np.cumsum(residual)
        cum_sum[-1] = 1. 
        indexes[k:N] = np.searchsorted(cum_sum, rand(N-k))
        
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
        
    def multinomial_resample(self):
        '''
        Multinomial resample of particles
        
        DESCRIPTION
        Compute the cumulativesum of the normalized weights.
        To select a weight we generate a random number uniformly 
        selected between 0 and 1 and use binary search to find
        its position inside the cumulative sum array.
        
        '''
        cum_sum = np.cumsum(self.weights)
        cum_sum[-1] = 1. # avoid round-off errors
        
        indexes= np.searchsorted(cum_sum, rand(len(self.weights)))
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
    
    
    def stratified_resample(self):
        '''
        Stratified resample of particles
        
        DESCRIPTION
        It works by dividing the cumulative sum into N equal sections, and 
        then selects one particle randomly from each section. This guarantees 
        that each sample is between 0 and 2/N 
        
        '''

        
        N = len(self.weights)
        # make N subdivisions, chose a random position within each one
        positions = (rand(N) + range(N)) / N
    
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
    
    def systematic_resample(self):
        '''
        Systematic_resample resample of particles
        
        DESCRIPTION
        As with stratified resampling the space is divided into N divisions
        We then choose a random offset to use for all of the divisions, 
        ensuring that each sample is exactly 1/N apart.

        '''
        
        N = len(self.weights)
    
        # make N subdivisions, choose positions 
        # with a consistent random offset
        positions = (np.arange(N) + random()) / N
    
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))
        
        
if __name__=="__main__":    
    """Executed only when the file is run as a script."""
    
    np.random.seed(123)
    
    
    x_varCollection=[]
    y_varCollection=[]
    duration=[]
    
    for N in [50,100,150,200,250]:
    
        control = pd.read_csv("Particles_2020/u.csv", header=None)
        radar = pd.read_csv("Particles_2020/meausure.csv", header=None)
        
        Q=[0.05,0.2]
        R=[1,0.2]
        
        #N=50
        start = time.time()
        pf=ParticleFilter(N,Q,R,[0,0])
        plot_particles=True
        
        x_mean = []
        var_mean = []
        
        plt.ion()
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot([], [])
        numOfResample=0
        activeP=[]
        

        for i in range(0,126):
            
            u=np.array(control.iloc[i,:])
            z=np.array(radar.iloc[i,:])
        
            pf.predict(u)
            pf.update(z)
    
            if neff(pf.weights)< (0.75)*N:
                
                activeP.append(neff(pf.weights))
                numOfResample=numOfResample+1
                

                pf.residual_resample()

                
            mu, var = pf.estimate()
            
            
            x_mean.append(mu)
            var_mean.append(var)
            
            if plot_particles:
                plt.scatter(pf.particles[:, 0],pf.particles[:, 1], 
                            color='k', marker=',', s=1)
            
            
            plt.scatter(mu[0], mu[1], marker='o',color='g',zorder=i)
            
            
            plt.axis((-2,3,-1,12))
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            
        x_mean=np.array(x_mean)
        var_mean=np.array(var_mean)
        
        x_varCollection.append(var_mean[:,0])
        y_varCollection.append(var_mean[:,1])
        
        end = time.time()
        duration.append(end - start)
        print(str(end - start) ," sec")
        print(mu,var)
        print("Number of resamples : ",str(numOfResample))
        
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    line1, = ax.plot([], [])
    plt.boxplot(x_varCollection)
    
    ax1.set_xticklabels(['50', '100', '150', '200','250'])
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    line1, = ax.plot([], [])
    plt.boxplot(y_varCollection)
    
    ax2.set_xticklabels(['50', '100', '150', '200','250'])
 
    
  