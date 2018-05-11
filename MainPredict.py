import numpy
import pandas
import pickle

def ReLU(vector):
  result = numpy.copy(vector)
  for i in range(result.shape[0]):
    result[i] = max(0, result[i])
  return result

#load the model
model = numpy.load("model.npz")
layerweights = model["layerweights"]
layerbiases = model["layerbiases"]

#load data
testDataRaw = pandas.read_csv("predict.txt", delimiter='|')

#load the column names
columnFile = open("model.col", "rb")
trainColumnNames = pickle.load(columnFile)

#dummy out categorical variables into binary and convert to matrix
testDataDummied = pandas.get_dummies(testDataRaw.drop("Responder___", axis=1))
testColumnNames = testDataDummied.columns

#code borrowed from https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data
#--------------------------------begin borrowed code----------------------------------------
# Get missing columns in the training test
if bool(set( testColumnNames ) - set( trainColumnNames )):
  print("Error: Unknown column names found")
  exit(0)
missing_cols = set( trainColumnNames ) - set( testColumnNames )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    testDataDummied[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
testDataDummied = testDataDummied[trainColumnNames]
#----------------------------------end borrowed code----------------------------------------

testDataConverted = numpy.asmatrix(testDataDummied.as_matrix()).astype(int)

#convert test data to matrix
testData = testDataConverted[:,1:]

#scoring
results = open("results.txt")

for i in range(testData.shape[0]):
  #convert the current record to a column vector
  curRecord = numpy.transpose(testData[i])
  
  #compute output
  output = numpy.add(numpy.matmul(layerweights[0], curRecord), layerbiases[0])
  for k in range(1, len(layerweights)):
    output = numpy.add(numpy.matmul(layerweights[k], ReLU(output)), layerbiases[k])
  
  #save the result of the score
  results.write(testDataConverted[i, 0] ": " + ReLU(output)[1]/(ReLU(output)[0] + ReLU(output)[1]))