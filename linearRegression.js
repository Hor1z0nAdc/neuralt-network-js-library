export default class LinearReg {
    constructor(batchSize, numOfParams, options) {
        this.batchSize = batchSize
        this.numOfParams = numOfParams
        this.learningRate = options?.learningRate || 0.0001
        this.trainingIterations = options?.trainingIterations || 1000
        this.accuracy = options?.accuracy || 2
        this.cost = 100
        this.neuron = {weights: this.defWeights(), bias: 0}
    }   
    // constructor(batchSize, learningRate, numOfParams, accuracy = 2, trainingIterations = 1000) {
    //     this.batchSize = batchSize
    //     this.learningRate = learningRate
    //     this.trainingIterations = trainingIterations
    //     this.cost = 100
    //     this.accuracy = accuracy 
    //     this.numOfParams = numOfParams
    //     this.neuron = {weights: this.defWeights(), bias: 0}
    // }

    getSlope() {
        return this.neuron.weights.map(w => {return parseFloat(w.toFixed(this.accuracy))})
    }

    getIntercept() {
        return this.neuron.bias.toFixed(this.accuracy)
    }

    getLine() {
        let variables = ["x","y","z"]
        let func = "y = "
        for(let i = 0; i < this.neuron.weights.length; i++) {
            func += `${this.neuron.weights[i].toFixed(this.accuracy)} * ${variables[i]} + `
        }
        func += `${this.neuron.bias.toFixed(this.accuracy)}`
        return  func
    }

    getFunction() {
        let variables = ["x","y","z"]
        let func = ""
        for(let i = 0; i < this.neuron.weights.length; i++) {
            func += `${this.neuron.weights[i].toFixed(this.accuracy)} * ${variables[i]} +`
        }
        func += ` ${this.neuron.bias.toFixed(this.accuracy)}`
        return func
    }

    getCost() {
        return this.cost
    }

    SSMean(targetValues) {
        let mean = targetValues.reduce((a,b) => a + b, 0) / targetValues.length
        let ssMean = 0
        for(let i = 0; i < targetValues.length; i++) {
            ssMean += Math.pow((targetValues[i] - mean), 2) 
        }
        return ssMean
        
    }

    variationMean(targetValues) {
        let variation = this.SSMean(targetValues) / targetValues.length
        return variation
    }

    variationFit(targetValues) {
        return this.cost / targetValues.length
    }

    RSquare(targetValues) {
        let varMean = this.variationMean(targetValues)
        let varFit = this.variationFit(targetValues)
        return (varMean - varFit) / varMean
    }

    defWeights() {
        let weights = []
        for(let i = 0; i < this.numOfParams; i++) {
            weights.push(1)
        }
        return weights
    }

    train(inputs, targetValues, costSpan) {
        for(let i = 0; i < this.trainingIterations; i++) {
            const batchIndexes = this.getBatch(inputs.length)
            let predictedValues = []
            let cost = 0
            let biasD = 0
            let weightsD = Array(this.numOfParams).fill(0)

           for(let j = 0; j < this.batchSize; j++) {
                predictedValues.push(this.predict(inputs[batchIndexes[j]]))
               
                let currentBiasD = - 2 * (targetValues[batchIndexes[j]] - predictedValues[j])
                for(let k = 0; k < this.neuron.weights.length; k++) {
                    weightsD[k] +=  - 2 * this.neuron.weights[k] * (targetValues[batchIndexes[j]] - predictedValues[j])
                }
                biasD += currentBiasD
               
                cost += Math.pow((targetValues[batchIndexes[j]] - predictedValues[j]), 2)
            }

           this.cost = cost
           if(cost < 0.1) {
                break
           }
           else {
               this.gradientDescent(biasD, weightsD)
           }
        }

        console.log(this.RSquare(targetValues))
     
        costSpan.innerText = this.cost     
    }

    updateCostElement(costSpan) {
        costSpan.innerText = this.cost
        setTimeout(this.updateCostElement, 500)
    }

    gradientDescent(biasD, weightsD) {
        let stepSizeBias = biasD * this.learningRate
        this.neuron.bias = (this.neuron.bias - stepSizeBias)

        for(let i = 0; i < this.numOfParams; i++) {
           this.neuron.weights[i] -= weightsD[i] * this.learningRate
        }  
    }

    getBatch(length) {
        let currentBatch = []
        let randomIndex
        let isRedundant
        for(let i = 0; i < length; i++) {
            do {
                randomIndex = Math.floor(Math.random() * length)
                isRedundant = currentBatch.includes(randomIndex)
                if(!currentBatch.includes(randomIndex)) currentBatch.push(randomIndex)
            }
            while(isRedundant)
        }
        return currentBatch
    }

    predict(values) {
        let predictedValue = this.neuron.bias
        for(let i = 0; i < this.numOfParams; i++) {
           
            predictedValue +=  this.neuron.weights[i] * values[i]
        }   
        return predictedValue.toFixed(this.accuracy)
    }
}