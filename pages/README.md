Notes on CNN's: 

Most popularly used for image analysis

Similar to a typical multi layer perceptron with the addition of convolutional hidden layers

These layers transform the input to next layer as a convolution operation 

They detet patterns in images, each conv layer has a number of filters. Each filter detects patterns

One pattern could be edges, a filter could be an edge detector, or simple shapes for example. Often called geometric filters

Deeper layers can detect more sophisticated objects, eyes, arms, cats, dogs etc. 

Take MNIST digits for example
   first hiddne layer is a convolutional layerd

   the convolution is sort of like a dot product against successive n*n submatrices of the target 
   matrix, assigned to each cell of the result matrixsd
   (I guess similar to interpolation?)

Useful reference: [deeplizard - Convolutional Neural Networks (CNNSs) explained](https://deeplizard.com/learn/video/YRhxdVk_sIs)

From deeplizard demo: 

