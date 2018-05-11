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

LEARNING_RATE = 0.02
BATCH_SIZE = 32
LAYER_ARRAY = [200, 100, 2]

#load data
trainDataRaw = pandas.read_csv("training.txt", delimiter='|')

#dummy out categorical variables into binary and convert to matrix
trainDataDummied = pandas.get_dummies(trainDataRaw)
columnNames = trainDataDummied.columns
trainDataConverted = numpy.asmatrix(trainDataDummied.as_matrix()).astype(int)

#convert train data to matrix
trainLabels = trainDataConverted[:, 1]
trainData = trainDataConverted[:,2:]

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

while shouldTrain:
  order = numpy.random.choice(trainData.shape[0], size=trainData.shape[0], replace=False)
  for i in range(0, len(order) - BATCH_SIZE, BATCH_SIZE):
    #gradient sums (used for batches)
    biasGradientSUM = [0] * len(LAYER_ARRAY)
    weightGradientSUM = [0] * len(LAYER_ARRAY)

    for j in range(i, i + BATCH_SIZE):
      #progress bar
      percentComplete = int(j / trainData.shape[0] * 100)

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
  
  #Get a subset of test images to test accuracy
  indexes = numpy.random.choice(trainData.shape[0], size=1000, replace=False)
  
  #testing accuracy
  correct = 0
  wrong = 0
  
  for num in indexes:
    curRecord = numpy.transpose(trainData[num])
    output = numpy.add(numpy.matmul(layerweights[0], curRecord), layerbiases[0])
    for k in range(1, len(LAYER_ARRAY)):
      output = numpy.add(numpy.matmul(layerweights[k], ReLU(output)), layerbiases[k])
    if (ReLU(output)[1][0]/(ReLU(output)[0][0] + ReLU(output)[1][0]) > 0.95):
      if (trainLabels[num] == 1):
        correct += 1
      else:
        wrong += 1
      
  #if the accuracy was achieved or if we have taken over the maximum number of times, terminate
  if (input("Accuracy in 95th percentile: " + str(correct/(correct + wrong + 1)) + ". Stop here? (y/n): ") == "y"):
    shouldTrain = False

#save the final model to a .npz file
numpy.savez("model", layerweights=layerweights, layerbiases=layerbiases)

#save the column names used
columnFile = open("model.col", "wb")
pickle.dump(columnNames, columnFile)