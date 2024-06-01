"use strict";
import * as math from "./math.js";
import Layer from "./layer.js"
import * as Activation from "./activationFunctions.js"
import * as Cost from "./costFunctions.js";
import LinearReg from "./linearRegression.js";

const costSpan = document.getElementById("costSpan")

let vector1 = [1,1,1]
let vector2 = [2,2,2]
let matrix1 = [[1,1,3], [2,2,2]]
let matrix2 = [[1,1,1,], [2,2,2]]

let inputs = [[0.8,0.2,0.1],[0.1,0.4,0.7]]
let weights = [[0.5,0.7,0.3],
               [0.2,0.6,0.3],
               [0.2,0.9,0.8]]    
let biases = [1,1,1]    
let class_targets =  [[1,0,0],[0,0,1]]

const linearInput1 = [[1],[2],[5],[6],[9]]
const linearInput2 = [[1,0],[2,0],[5,0],[6,0],[9,0]]
const targetValues = [2,4,10,12,18]


// let layer1 = new Layer(3,2)
// let layer2 = new Layer(2,3, true)
// layer1.forward(matrix1)
// layer2.forward(layer1.output)
// Cost.categoricalCrossEntropy(layer2.output, class_targets)
// Cost.accuracy(inputs, class_targets)
// console.log(Cost.squaredError(inputs, class_targets))

const options = {
    learningRate: 0.0001,
    trainingIterations: 100000,
    accuracy: 3
}
let linear = new LinearReg(5, 1, options)
console.log("BEFORE\n")
console.log("intercept ", linear.getIntercept())
console.log("slope ", linear.getSlope())
console.log(linear.getLine())
linear.train(linearInput1, targetValues, costSpan)
console.log("\nAFTER\n")
console.log("intercept ", linear.getIntercept())
console.log("slope ", linear.getSlope())
console.log(linear.getLine())
console.log("prediction for 3:", linear.predict([3,0]))

functionPlot({
    target: '#chart',
    data: [
        { fn: linear.getFunction() },
        { points: [
            [linearInput1[0][0], targetValues[0]],
            [linearInput1[1][0], targetValues[1]],
            [linearInput1[2][0], targetValues[2]],
            [linearInput1[3][0], targetValues[3]],
            [linearInput1[4][0], targetValues[4]]
          ],
          fnType: 'points',
          graphType: 'scatter' }
    ]
  })