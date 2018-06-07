import sys
import numpy as np

def estimate_value(state):
   print(state)
   print(state.x)
   print(state.done)

   value = 0.0
   print(value)
   return value


def estimate_probabilities(state):
   print(state)

   prob = 0.25
   probabilities = np.array([i*0+prob for i in range(0,4)])
   return(probabilities)
