import numpy
import pandas
import pickle

def ReLU(vector):
  result = numpy.copy(vector)
  for i in range(result.shape[0]):
    result[i] = max(0, result[i])
  return result

def ReLUDerivative(vector):
  result = numpy.copy(vector)
  for i in range(result.shape[0]):
    if (result[i] > 0):
      result[i] = 1
    else:
      result[i] = 0
  return result

#load data
trainDataRaw = pandas.read_csv("training.txt", delimiter='|')

#set parameters
LEARNING_RATE = 0.02
BATCH_SIZE = 32
LAYER_ARRAY = [200, 100, 2]
VALIDATION_SIZE = int(len(trainDataRaw) / 3)

#dummy out categorical variables into binary and convert to matrix
trainDataDummied = pandas.get_dummies(trainDataRaw, drop_first=True)
columnNames = trainDataDummied.columns
trainDataConverted = numpy.asmatrix(trainDataDummied.as_matrix())

#convert train data to matrix
trainLabels = trainDataConverted[:, 1].astype(int)
trainData = trainDataConverted[:,2:].astype(float)

#normalize numerical data
maxVals = numpy.max(trainData, axis=0)
minVals = numpy.min(trainData, axis=0)

for i in range(trainData.shape[1]):
  if minVals[0,i] != 0 or maxVals[0,i] != 1:
    trainData[:,i] = numpy.divide(trainData[:, i] - minVals[0,i], maxVals[0,i] - minVals[0,i])

#Advanced classifier

#training
#initialize the layers
layerNum = 0
layerweights = [0] * len(LAYER_ARRAY)
layerbiases = [0] * len(LAYER_ARRAY)
for num in LAYER_ARRAY:
  if layerNum == 0:
    layerweights[layerNum] = 0.01 * numpy.random.randn(num, trainData.shape[1])
  else:
    layerweights[layerNum] = 0.01 * numpy.random.randn(num, layerweights[layerNum - 1].shape[0])
  layerbiases[layerNum] = numpy.zeros((num, 1))
  layerNum = layerNum + 1

#initialize activations array
activations = [0] * len(LAYER_ARRAY)

#gradients
biasGradient = [0] * len(LAYER_ARRAY)
weightGradient = [0] * len(LAYER_ARRAY)
activationDerivatives = [0] * len(LAYER_ARRAY)

shouldTrain = True

#number of epochs
numEpochs = 0

#get validation set
order = numpy.random.choice(trainData.shape[0], size=trainData.shape[0], replace=False)
validationSet = order[0:VALIDATION_SIZE]
order = order[VALIDATION_SIZE:]

while shouldTrain:
  numpy.random.shuffle(order)
  for i in range(0, len(order) - BATCH_SIZE, BATCH_SIZE):
    #gradient sums (used for batches)
    biasGradientSUM = [0] * len(LAYER_ARRAY)
    weightGradientSUM = [0] * len(LAYER_ARRAY)

    for j in range(i, i + BATCH_SIZE):
      #progress bar
      percentComplete = int(j / len(order) * 100)

      print("-" * percentComplete + " " + str(percentComplete) + "%", end="\r")

      #convert the current record to a column vector
      curRecord = numpy.transpose(trainData[order[j]])

      #compute output
      activations[0] = numpy.add(numpy.matmul(layerweights[0], curRecord), layerbiases[0])
      for k in range(1, len(LAYER_ARRAY)):
        activations[k] = numpy.add(numpy.matmul(layerweights[k], ReLU(activations[k - 1])), layerbiases[k])

      #expected output (all elements 0 except the correct output)
      outputExpected = numpy.zeros((LAYER_ARRAY[len(LAYER_ARRAY) - 1], 1))
      outputExpected[trainLabels[order[j]]] = 1

      for k in reversed(range(0, len(LAYER_ARRAY))):
        #compute the gradient of the previous layer's weights
        if k == len(LAYER_ARRAY) - 1:
          #compute the expected minus the output
          activationDerivatives[k] = numpy.subtract(ReLU(activations[k]), outputExpected)
        else:
          #backpropagate
          activationDerivatives[k] = numpy.matmul(numpy.transpose(layerweights[k + 1]), biasGradient[k + 1])

        ReLUDerivatives = ReLUDerivative(activations[k])
        biasGradient[k] = numpy.multiply(activationDerivatives[k], ReLUDerivatives)
        if (k == 0):
          weightGradient[k] = numpy.matmul(biasGradient[k], numpy.transpose(curRecord))
        else:
          weightGradient[k] = numpy.matmul(biasGradient[k], numpy.transpose(ReLU(activations[k-1])))

        #sum up the gradient
        biasGradientSUM[k] = biasGradientSUM[k] + biasGradient[k]
        weightGradientSUM[k] = weightGradientSUM[k] + weightGradient[k]

    #subtract the gradients from the neurons
    for j in range(len(LAYER_ARRAY)):
      layerweights[j] = numpy.subtract(layerweights[j], LEARNING_RATE * weightGradientSUM[j])
      layerbiases[j] = numpy.subtract(layerbiases[j], LEARNING_RATE * biasGradientSUM[j])
  
  numEpochs = numEpochs + 1
  print("Epoch " + str(numEpochs) + " Complete")
  print("Testing accuracy")
  
  #testing accuracy
  evalMatrix = numpy.zeros((VALIDATION_SIZE, 2))
  evalNum = 0
  
  for num in validationSet:
    curRecord = numpy.transpose(trainData[num])
    output = numpy.add(numpy.matmul(layerweights[0], curRecord), layerbiases[0])
    for k in range(1, len(LAYER_ARRAY)):
      output = numpy.add(numpy.matmul(layerweights[k], ReLU(output)), layerbiases[k])
      
    #add to evaluation matrix
    evalMatrix[evalNum][0] = ReLU(output)[1][0]/(ReLU(output)[0][0] + ReLU(output)[1][0])
    evalMatrix[evalNum][1] = trainLabels[num]
    evalNum += 1
  
  #check distribution along percentiles
  evalMatrix = evalMatrix[(-evalMatrix[:,0]).argsort()]
  print("Total responders in validation set: " + str(numpy.sum(evalMatrix, axis=0)[1]) + "\n")
  for i in range(20):
    print("Responders in " + str(100 - (i + 1) * 5) + "th percentile: " + str(numpy.sum(evalMatrix[int(i * (VALIDATION_SIZE/20)):int((i + 1) * (VALIDATION_SIZE/20)), 1])) + "\n")
  
  
  #if the accuracy was achieved or if we have taken over the maximum number of times, terminate
  if (input("Stop here? (y/n): ") == "y"):
    shouldTrain = False

#save the final model to a .npz file
numpy.savez("model", layerweights=layerweights, layerbiases=layerbiases)

#save the column names used
columnFile = open("model.col", "wb")
pickle.dump(columnNames, columnFile)