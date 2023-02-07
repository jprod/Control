import lqr
import numpy as np

'''
adapted from:
Author: Addison Sears-Collins
https://automaticaddison.com
Description: Linear Quadratic Regulator example 
(two-wheeled differential drive robot car)
'''

'''
Example inputs to work for this set up
A = np.array([  [1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, 0, 1.0]])
 
R = np.array([[0.01, 0],  
                [0, 0.01]]) 

Q = np.array([[0.639, 0, 0], 
                [0, 1.0, 0],  
                [0, 0, 1.0]]) 

'''


def world(A, B, Q, R, time, dt): 

    sensory = np.array([0,0,0]) 
    reference = np.array([2.000,2.000,np.pi/2])  
                   
    for i in range(time):
        print(f'iteration = {i} seconds')
        print(f'Current State = {sensory}')
        print(f'Desired State = {reference}')
         
        error = sensory - reference
        error_magnitude = np.linalg.norm(state_error)     
        print(f'State Error Magnitude = {error_magnitude}')
        
        # action matrix
        B = getB(sensory[2], dt)
         
        # effector
        optimal_control_input = lqr(sensory, reference, Q, R, A, B) 
         
        # this is what the forward model will get to estimate future states
        print(f'Control Input = {optimal_control_input}')
        
        # output of the system / plant  (feedback)
        sensory = state_space_model(A, sensory, B, optimal_control_input)  
 
        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 0.01:
            print("\nGoal Has Been Reached Successfully!")
            break
             
        print()
 
world()