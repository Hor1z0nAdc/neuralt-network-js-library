import NeuralNetwork from "./NeuralNetwork.js";
import * as cost from "./costFunctions.js"
import * as activation from  "./activationFunctions.js"
import * as optimal from  "./optimizers.js"

const multyevenDataSet = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,1,1],[1,0,0,0],[1,1,1,0],[1,1,1,1]]
const multyevenTargetClass = [[0,0],[0,1],[0,0],[0,1],[0,1],[0,0],[0,0],[0,1]] 
const evenDataSet = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,1,1],[1,0,0,0],[1,1,1,0],[1,1,1,1]]
const evenTargetClass = [0,1,0,1,1,0,0,1] 
const peopleDataSet = [[175,65], [140,35], [166,50], [180,80], [173,70], [160,48], [145, 40], [138, 34], [172,52], [185,85]]
const peopleTargetClass = [2, 0 ,1, 2, 2, 1, 0, 0, 1, 2]
const widthDataSet = [[10], [8],[5],[12],[3],[6],[9],[15],[14],[18]]
const widthTarget = [[7, 3], [5.6, 2.4], [3.5, 1.5], [8.4, 3.6], [2.1, 0.9], [4.2, 1.8], [6.3, 2.7], [10.5, 4.5], [9.8, 4.2], [12.6, 5.4]]
const regressionDataSet = [[0], [1], [3], [4], [8], [9], [10], [15], [16], [19]]
const regressionTargetClass = [0,2,6,8,16,18,20,30,32,38]
const squareDataSet = [[0], [1], [3], [4], [8], [10], [11], [9], [13],]
const squareTargetClass = [0, 1, 9, 16, 64, 100, 121, 81, 169]
//z = x^2 + y
const equationDataSet = [[2,0], [2,1], [3,0], [3,1], [4,1], [4,0], [6,2], [6,1], [1,0], [1,1], ] 
const equationTargetClass = [4,5,9,10,17,16,38,37,1,2]

//single output classification
const options = {
    outputActivationF: activation.sigmoid,
    batchSize: 1,
    learningRate: 0.01,
    trainingIterations: 1000000,
    isClassification: false,
    maxError: 0.000001
}

let nn = new NeuralNetwork(activation.ReLU, cost.meanSquaredError, options)
nn.addLayer(4,4)
nn.addLayer(4,6)
nn.addLayer(6,2, true)

console.log(nn.predict([[0,1,1,0]]))
console.log("BEFORE", nn.test(multyevenDataSet, multyevenTargetClass))
nn.train(multyevenDataSet, multyevenTargetClass)
console.log("AFTER",nn.test(multyevenDataSet, multyevenTargetClass))
console.log(nn.predict([[1,1,1,0]]))

//Fix cross entropy
//When learning rate is low sometimes everything becomes NaN - because weights are jumping too much and might beome infinit