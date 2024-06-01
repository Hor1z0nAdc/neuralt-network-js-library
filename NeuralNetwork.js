import Layer from "./layer.js"
import { accuracyClass, meanError } from "./costFunctions.js"
import { decay } from "./optimizers.js"

export default class NeuralNetwork {
    constructor(activationF, costF, options) {
        this.layers = []
        this.cost = 0
        this.trainingAccuracy = -1
        this.trainingTime = -1
        this.testAccuracy = -1
        this.echoes = 0

        this.activationF = activationF instanceof Function ? activationF() : null
        this.costF = costF instanceof Function ? costF() : null
        this.outputActivationF =  options?.outputActivationF ?  options.outputActivationF() : this.activationF
        this.trainingOActivationF = options?.trainingOActivationF ? options.trainingOActivationF() : this.outputActivationF
        this.testOActivationF = options?.testOActivationF ? options.trainingOActivationF() : this.outputActivationF
        this.optimizer = typeof options?.optimizer === 'function' ? options.optimizer : null

        this.batchSize = options?.batchSize || 2
        this.decay = options?.decay ? options.decay : undefined
        this.momentum = options?.momentum ? options.momentum : undefined
        this.learningRate = options?.learningRate || 0.0001
        this.currentLearningRate = this.learningRate
        this.maxError = options?.maxError || 0.1
        this.trainingIterations = options?.trainingIterations || 1000
        this.isClassification = options?.isClassification == false ? false : true 
    }

    addLayer(n_inputs, n_neurons, output_layer = false) {
        this.layers.push(new Layer(n_inputs, n_neurons, output_layer))
    }

    getLayerWeights(layer_index) {
        return this.layers[layer_index].getWeights()
    }

    getLayerBiases(layer_index) {
        return this.layers[layer_index].getBiases()
    }

    getLayerNum() {
        return this.layers.length
    }

    getLayerNeuronNum(layer_index) {
        return this.layers[layer_index].getNumOfNeurons() 
    }

    
    getLayerInputNum(layer_index) {
        return this.layers[layer_index].getNumOfInputs() 
    }

    getAllLayers() {
        let layerNeurons = []
        layerNeurons.push(this.layers[0].weights[0].length)
        
        for(let i = 0; i < this.layers.length; i++) {
           layerNeurons.push(this.layers[i].biases.length)
        }
        return layerNeurons
    }

    getTrainingAccuracy() {
        return this.trainingAccuracy
    }

    getTestAccuracy() {
        return this.testAccuracy
    }

    getCost() {
        return this.cost
    }

    getLayerOutput(layerIndex) {
        return this.layers[layerIndex].output
    }

    getLayerRawOutput(layerIndex) {
        return this.layers[layerIndex].rawOutput
    }

    train(inputs, targetClass) {
        let randomBatch
        let predictedValues
        
        let firstDate = new Date()
        for(let i = 0; i < this.trainingIterations; i++) {
            //Get a random batch of inputs based on batch size
            randomBatch = this.getBatch(inputs, targetClass)

            //Calculate the predicted values and the cost based on the values
            predictedValues = this.calcPrediction(randomBatch.batch, this.trainingOActivationF)
            this.cost = this.costF.calc(predictedValues, randomBatch.targetClassBatch)
            
            //determine if there was an error in the prediction
            let isError = this.#isTrainingError(predictedValues)
            
            //return if the training stops due to error or max. limit reach
            let object = this.#trainingEnd(isError, i)
            if(object != null) {
                this.#finishEnd(object, firstDate, predictedValues, randomBatch)
                return object
            }
            
            //Optimize weights and biases via backpropogation
            this.backProp(randomBatch.targetClassBatch, i)
        }
   }

   test(inputs, targetClass) {
        const predictedValues = this.calcPrediction(inputs, this.testOActivationF)
        this.cost = this.costF.calc(predictedValues, targetClass)

        if(this.isClassification) {
            this.testAccuracy = accuracyClass(predictedValues, targetClass)
        }
        else {
            this.testAccuracy = meanError(predictedValues, targetClass)
        }
        return { accuracy: this.testAccuracy, cost: this.cost }
   }

    backProp(target, iteration) {
        //Iterate through all the layers in reverse order and apply suitable backprop algortihm
        for (let i = this.layers.length - 1; i >= 0; i--) {
           if(i == this.layers.length - 1) {
                this.layers[i].backPropOutput(this.costF.calcD, this.outputActivationF.calcD, target, this.isClassification)
           }
           else {
            this.layers[i].backPropHidden(this.layers[i+1].gamma, this.layers[i+1].weights, this.activationF.calcD, target)
           }
        }

        //If decay is set apply it to change the current l. rate based on the current iteration
        if(this.decay && !this.optimizer) {
            this.currentLearningRate = decay(iteration, this.currentLearningRate, this.decay)
        }
    
        //update weights and biases on every layer
        for(let i = 0; i < this.layers.length; i++) {
            this.layers[i].updateParams(this.currentLearningRate, this.optimizer, this.momentum, iteration)
        }
    }

    predict(inputs) {
        let length = this.layers.length
        return this.networkForward(inputs, length)
    }

    calcPrediction(inputs, outputActivationF) {
        let length = this.layers.length
        return this.networkForward(inputs, length, outputActivationF)
    }
    
    networkForward(input, length, outputActivationF) {
        //Give the input to the first layer 
        //If the function isn't called from a training or testing function it will pass
        //the general output activation function to the forward layer function
        outputActivationF = outputActivationF == undefined ? this.outputActivationF : outputActivationF
        this.layers[0].forward(input, this.activationF, outputActivationF) 
        
        //Calc output for the remaining layers
        for(let i = 1; i < length; i++) {
           this.layers[i].forward(this.layers[i-1].output, this.activationF, outputActivationF)
        }

        return this.layers[length-1].output
    }

    getBatch(inputs, targetClass) {
        let batchIndex = []
        let batch = []
        let targetClassBatch = []
        let randomIndex
        let isRedundant
        for(let i = 0; i < this.batchSize; i++) {
            do {
                //Only push index to the batch if it isn't there yet
                randomIndex = Math.floor(Math.random() * inputs.length)
                isRedundant = batchIndex.includes(randomIndex)
                if(!batchIndex.includes(randomIndex)) {
                    batchIndex.push(randomIndex)
                    batch.push(inputs[randomIndex])
                    targetClassBatch.push(targetClass[randomIndex])
                }
            }
            while(isRedundant)
        }
        return { batch, targetClassBatch}
    }

    loadNetwork(data) {
        let length = data["layers"].length
        data["layers"].forEach((layer,index) => {
            let isOutputLayer = length - 1 == index 
            this.addLayer(layer.numOfInputs, layer.numOfNeurons, isOutputLayer)
            this.layers[index].weights = layer.weights
            this.layers[index].biases = layer.biases
        });

        this.activationF = data.activationF
        this.costF = data.costF
        this.trainingOActivationF = data.trainingOActivationF
        this.testOActivationF = data.testOActivationF
        this.batchSize = data.batchSize
        this.learningRate = data.learningRate
        this.trainingIterations = data.trainingIterations
        this.maxError = data.maxError
        this.trainingAccuracy = data.trainingAccuracy
    }

    createNetworkData() {
        let dataObj = {}
        let arrayOfLayers = []
        for(let i = 0; i < this.layers.length; i++) {
            const weights =  this.getLayerWeights(i)
            const biases = this.getLayerBiases(i)

            let obj= { weights:weights, 
                                 biases: biases, 
                                 numOfNeurons: this.getLayerNeuronNum(i),
                                 numOfInputs: this.getLayerInputNum(i)}
            arrayOfLayers.push(obj)                         
        }
        dataObj["layers"] = arrayOfLayers
        dataObj["activationF"] = this.activationF
        dataObj["costF"] = this.costF
        dataObj["trainingOActivationF"] = this.trainingOActivationF
        dataObj["testOActivationF"] = this.testOActivationF
        dataObj["batchSize"] = this.batchSize
        dataObj["learningRate"] = this.learningRate
        dataObj["trainingIterations"] = this.trainingIterations
        dataObj["maxError"] = this.maxError
        dataObj["trainingAccuracy"] = this.trainingAccuracy
        return dataObj
    }

    #trainingEnd(isError, i) {
        let isEnd = false
        let object = {}
        
        //When an error occured
        if(isError) {
            isEnd = true
            object["isError"] = true
        }

        //When there is no error check if maxError or max iteration has been reached
        else {
            object["isError"] = false

            if(this.cost <= this.maxError) {
                isEnd = true
                object["isTargetError"] = true
               
            }
    
            if(i == this.trainingIterations - 1) {
                isEnd = true
                object["isTargetError"] = false
            }
        }
        
        //When training stops due to either error or reach limit return the constructed object
        if(isEnd) {
            this.echoes = i + 1
            object["echoNum"] = this.echoes
            
            return object
        }
        else {
            return null
        }
    }

    #isTrainingError(predictedValues) {
        let isError

        if(predictedValues[0].length == undefined) {
            isError = predictedValues.some(element => isNaN(element))
        }
        else {
            predictedValues.every(vector => {
                isError = vector.some(element => isNaN(element))

                if(isError) return false
                return true
            })
        }

        return isError
    }

    #finishEnd(object, firstDate, predictedValues, randomBatch) {
        //Calc. the accuracy after the training is done.
        if(this.isClassification) {
            this.trainingAccuracy = accuracyClass(predictedValues, randomBatch.targetClassBatch)
        }
        else {
            this.testAccuracy = meanError(predictedValues, randomBatch.targetClassBatch)
        }

        //determine elapsed time between the start and the end of training in milliseconds
        let secondDate = new Date()
        object["trainingTime"] = secondDate - firstDate
    }
}