import * as math from "./math.js"

export default class Layer {
    constructor(n_inputs, n_neurons, isOutputLayer) {
        this.weights = randomWeights(n_inputs, n_neurons)
        this.weightsD = Array.from(Array(n_neurons), () => new Array(n_inputs).fill(0))
        this.biasesD = Array(n_neurons).fill(0)
        this.numOfNeurons = n_neurons
        this.numOfInputs = n_inputs
        this.gamma = Array(n_neurons)
        this.biases = Array(n_neurons).fill(0)
        this.inputs = null
        this.rawOutput = []
        this.output = []
        this.isOutputLayer = isOutputLayer
        this.G = Array.from(Array(n_neurons), () => new Array(n_inputs).fill(0))
        this.Gb = Array(n_neurons).fill(0)
        this.change = Array.from(Array(n_neurons), () => new Array(n_inputs).fill(0))
        this.changeB = Array(n_neurons).fill(0)
    }
    
    getBiases() {
        return this.biases
    }
    
    getWeights() {
        return this.weights
    }
    
    getOutput() {
        return this.output
    }

    getNumOfNeurons() {
        return this.numOfNeurons
    }

    getNumOfInputs() {
        return this.numOfInputs
    }
    
    forward(inputs, activationF, outputActivationF) {
        //Save inputs for calculating derivative during backpropogation
        this.inputs = inputs
    
        //Calculate the weights * input based on matrix multiplication
        let mulResult = math.matrixMul(inputs, math.transpose(this.weights))
        
        //Add the biases to the result of weights * input (= x for activation function)
        for(let i = 0; i < mulResult.length; i++) {
            for(let j = 0; j < mulResult[0].length; j++) {
                mulResult[i][j] += this.biases[j]
            }
        } 
        this.rawOutput = mulResult
        
        //Determine final output by the activation function
        let activationResult 
        if(!this.isOutputLayer){
            activationResult = activationF.calc(mulResult)
        }
        else {
            activationResult = outputActivationF.calc(mulResult)
        }
        this.output = activationResult 
    }

    backPropOutput(costD, activationD, target, isClass) {
        
        //Calc. gamma, which equals derivative of cost function * derivative of activation function
        //for every output neuron
        for(let i = 0; i <this.output[0].length ; i++) {
            //if the given problem is a classification problem
            if(isClass) {

                //calc. the average derivative of the activation f. on a given batch of samples
                let avgActivationD = 0

                for(let j = 0; j < this.output.length; j++) {
                    avgActivationD += activationD(this.output[j], i)
                }
                avgActivationD /= this.output.length

                this.gamma[i] = costD(this.output, target, i) * avgActivationD
            }

            //if the given problem is a regression problem
            else {
                this.gamma[i] = costD(average(this.output, i), average(target, i)) * activationD(average(this.output, i))
            }
            
        }  
 
        //Calc. delta weights based on gamma * the input, that we multiplied with the current weight
        //Use the average of the inputs, because the input is actually a batch of inputs
        for(let i = 0; i <this.weights.length ; i++) {
            for(let j = 0; j <this.weights[0].length ;j++) {
                this.weightsD[i][j] = this.gamma[i] * average(this.inputs,j)
            }
            this.biasesD[i] = this.gamma[i]
        }     
    }

    backPropHidden(gammaForward, weightsForward, activationD) {
        //Calc. the gamma, where the output is a 2d array of activation output
        for(let i = 0; i <this.output[0].length ; i++) {
            this.gamma[i] = 0
            

            //add to gamma the multiplied by corresponding weight value of the forward layer
            for(let j = 0; j <gammaForward.length ; j++) {
                this.gamma[i] += gammaForward[j] * weightsForward[j][i]
                
            }

            //finally multiple the gamma by the derivative of the activation function
            this.gamma[i] *= activationD(average(this.output, i))
        }
    
        //Calc the delta weights and delta biases
        for(let i = 0; i < this.weights.length; i++) {
            for(let j = 0; j < this.weights[0].length; j++) {
                this.weightsD[i][j] = this.gamma[i] * average(this.inputs,j)
            }
            this.biasesD[i] = this.gamma[i]
        }
    }

    updateParams(learningRate, optimizer, momentum) {
        let currentLRate = learningRate

        if(optimizer) {
            optimizer.prep(this.G, this.Gb, this.weightsD, this.biasesD, this.change)
            optimizer.calc(currentLRate, this.weights, this.biases, this.weightsD, this.biasesD, this.G, this.Gb, this.change, this.changeB)
        }
        else {
            for(let i = 0; i <this.weights.length ; i++) {
                for(let j = 0; j <this.weights[0].length ; j++) {
                    this.weights[i][j] -= this.weightsD[i][j] * currentLRate
                }

                this.biases[i] -= this.biasesD[i] * learningRate 
            }
        }

        for(let i = 0; i <this.weights.length ; i++) {
            for(let j = 0; j <this.weights[0].length ; j++) {

                // if there is an optimizer utilize it to optimize learning rate and update parameters
                if(momentum) {
                    let newChange = momentum.calc(currentLRate, this.weightsD[i][j], this.change[i][j])
                    this.change[i][j] = newChange
                    this.weights[i][j] -= newChange
                    break
                }
                else if(optimizer) {
                    this.G[i][j] = optimizer.prep(this.G[i][j], this.weightsD[i][j])
                    currentLRate = optimizer.calc(this.learningRate, this.G[i][j])
                }
                this.weights[i][j] -= this.weightsD[i][j] * currentLRate
            }
            this.biases[i] -= this.biasesD[i] * learningRate 
        }
    }
}

function randomWeights(n_inputs, n_neurons) {
    let output= []
    for(let i = 0; i < n_neurons; i++) {
        let vector = []
        for(let j = 0; j < n_inputs; j++) {
            vector.push(Math.random() - 0.5)
        }
        output.push(vector)
    }
    return output
}


export function average(inputs,j) {
    let sum = 0
    
    //If output is only 1 sample
    if(inputs[0].length == undefined) {
        return inputs[j]
    }
    
    //If output is an actual batch of samples
    else {
        for(let i = 0; i < inputs.length; i++) {
            sum += inputs[i][j] 
        }
        return sum / inputs.length
    }
}   
