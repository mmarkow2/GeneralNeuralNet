import numpy
import pandas
import pickle

#load data
testDataRaw = pandas.read_csv("predict.txt", delimiter='|')

#load the model
model = numpy.load("model.npz")
layerweights = model["layerweights"]
layerbiases = model["layerbiases"]

#load the column names
columnFile = open("model.col", "rb")
trainColumnNames = pickle.load(columnFile)

#dummy out categorical variables into binary and convert to matrix
testDataDummied = pandas.get_dummies(testDataRaw)
testColumnNames = testDataDummied.columns

#code borrowed from https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data
#--------------------------------begin borrowed code----------------------------------------
# Get missing columns in the training test
if (set( testColumnNames ) - set( trainColumnNames ) is not None):
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
testData = testDataConverted[:,2:]

#testing
correct = 0
wrong = 0

for i in range(testData.shape[0]):
  #convert the current record to a column vector
  curRecord = numpy.transpose(testData[i])
  
  #compute output
  output = numpy.add(numpy.matmul(layerweights[0], curRecord), layerbiases[0])
  for k in range(1, len(LAYER_ARRAY)):
    output = numpy.add(numpy.matmul(layerweights[k], ReLU(output)), layerbiases[k])
  
  guess = ReLU(output).argmax()
    
  #add to stats
  if (guess == testLabels[i]):
    correct += 1
  else:
    wrong += 1
    
print("Accuracy: ")
print(correct / (correct + wrong))
