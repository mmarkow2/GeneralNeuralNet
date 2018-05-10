#testing
correct = 0
wrong = 0

for i in range(testImages.shape[0]):
  #convert the current image to a column vector
  curImage = numpy.transpose(testImages[i])
  
  #compute output
  output = numpy.add(numpy.matmul(layerweights[0], curImage), layerbiases[0])
  for k in range(1, len(LAYER_ARRAY)):
    output = numpy.add(numpy.matmul(layerweights[k], ReLU(output)), layerbiases[k])
  
  guess = ReLU(output).argmax()
  
  #display number and guess
  print(mndata.display(testImagesInput[i]))
  print("Guess: ")
  print(guess)
    
  #add to stats
  if (guess == testLabels[i]):
    correct += 1
  else:
    wrong += 1
    
print("Accuracy: ")
print(correct / (correct + wrong))
