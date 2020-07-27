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
Optionally, compute weighted mean and covariance of the set of particles to get a state estimate. 
