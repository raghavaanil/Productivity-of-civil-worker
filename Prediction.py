import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))


## Building a Predictive System 
input_data = (37,0,9,4,2512,0,169,98.5,9.92,15.5,-224.34,6,2,8.0,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) 

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not productive')
else:
  print('The person is productive')