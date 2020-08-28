# Particle_Filter
Particle filter for estimating x, y of a robot

##  Generic Particle Filter Algorithm
1. **Randomly generate a bunch of particles.**<br>
Particles can have position, heading, and/or whatever other state variable you need to estimate. Each has a weight (probability) indicating how likely it matches the actual state of the system. Initialize each with the same weight.
2. **Predict next state of the particles** <br>
Move the particles based on how you predict the real system is behaving.
3. **Update**<br>
Update the weighting of the particles based on the measurement. Particles that closely match the measurements are weighted higher than particles which don’t match the measurements very well.
4. **Resample**<br>
Discard highly improbable particle and replace them with copies of the more probable particles.
5. **Compute Estimate**<br>
Compute weighted mean and covariance of the set of particles to get a state estimate. 

## Task 
- The system of two states X and Y is described by
following equations :
x_k=0.3x_k+e^(-y_(k-1)^2 )+u_(1,k-1)   (1)
y_k=0.3x_k+e^(-x_(k-1)^2 )+u_(2,k-1)  (2)

- The system is observed by a radar with the measuring model being described by the following equation:
d_k=√(x_k^2+y_k^2 )+e_(1,k)  (3)

- e1 ~ N(0,1)
- e2 ~ N(0,0.2)
- u1 ~ N(0,0.05)
- u2 ~ N(0.2)

Do the best estimate for the state X,Y
