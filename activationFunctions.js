export function linear() {
    let calc = (input) => {
        return input
    }

    let calcD = () => {
        return 1
    }

    return { calc, calcD }
}

export function binaryStep() {
    let calc = (input) => {
        let activatedOuput = []

        //If the batch size is only 1
        if(input[0].length == undefined) {
            for(let i = 0; i < input.length; i++) {
                if(input[i] > 0) activatedOuput.push(1) 
                else  input[i] = activatedOuput.push(0)
            }
        }
        //If there is an actual batch of samples
        else {
            for(let i = 0; i < input.length; i++) {
                let vector = []

                for(let j = 0; j < input[0].length; j++) {
                   if(input[i][j] > 0) vector.push(1)
                   else  vector.push(0)
                }
                activatedOuput.push(vector)
            }
        }
        return activatedOuput
    }

    let calcD = () => {
        return 0
    }

    return { calc, calcD }
}

export function sigmoid() {
    let calc = (input) => {
        let activatedOuput = []

        //If the batch size is only 1
        if(input[0].length == undefined) {
            for(let i = 0; i < input.length; i++) {
                activatedOuput.push(1/(1 + Math.pow(Math.E, -input[i])))
            }
        }
        //If there is an actual batch of samples
        else {
            for(let i = 0; i < input.length; i++) {
                let vector = []

                for(let j = 0; j < input[0].length; j++) {
                   vector.push(1/(1 + Math.pow(Math.E, -input[i][j])))
                }
                activatedOuput.push(vector)
            }
        }
        return activatedOuput
    }

    let calcD = (output) => {
       return output * (1 - output)
    }

    return { calc, calcD }
}

export function ReLU() {
    let calc = (input) => {
        let activatedOuput = []

        //If the batch size is only 1
        if(input[0].length == undefined) {
            for(let i = 0; i < input.length; i++) {
                activatedOuput.push(Math.max(0, input[i]))
            }
        }
        //If there is an actual batch of samples
        else {
            for(let i = 0; i < input.length; i++) {
                let vector = []

                for(let j = 0; j < input[0].length; j++) {
                   vector.push(Math.max(0, input[i][j]))
                }
                activatedOuput.push(vector)
            }
        }
        return activatedOuput
    }

    let calcD = (x) => {
        if(x >= 0) return 1
        else return 0
    }
    
    return { calc, calcD }
}

export function leakyReLU(alpha = 0.01) {
    let calc = (input) => {
        let activatedOuput = []

        //If the batch size is only 1
        if(input[0].length == undefined) {
            for(let i = 0; i < input.length; i++) {
                activatedOuput.push(Math.max(alpha * input[i], input[i]))
            }
        }
         //If there is an actual batch of samples
        else {
            for(let i = 0; i < input.length; i++) {
                let vector = []

                for(let j = 0; j < input[0].length; j++) {
                   vector.push(Math.max(alpha * input[i][j], input[i][j]))
                }
                activatedOuput.push(vector)
            }
        }
        return activatedOuput
    }

    let calcD = (x) => {
        if(x >= 0) return 1
        else return alpha
    }
    
    return { calc, calcD }
}

export function Tanh() {
    let calc = (input) => {
        const E = Math.E
        let activatedOuput = []

        //If the batch size is only 1
        if(input[0].length == undefined) {
            for(let i = 0; i < input.length; i++) {
                const numerator = Math.pow(E, input[i]) - Math.pow(E, -input[i])
                const denominator = Math.pow(E, input[i]) + Math.pow(E, -input[i])
                activatedOuput.push(numerator / denominator)
            }
        }
          //If there is an actual batch of samples
        else {
            for(let i = 0; i < input.length; i++) {
                let vector = []

                for(let j = 0; j < input[0].length; j++) {
                    const numerator = Math.pow(E, input[i][j]) - Math.pow(E, -input[i][j])
                    const denominator = Math.pow(E, input[i][j]) + Math.pow(E, -input[i][j])
                    vector.push(numerator / denominator)
                }
                activatedOuput.push(vector)
            }
        }
        return activatedOuput
    }

    let calcD = (input) => {
        return 1 - (input * input)
    }

    return { calc, calcD}
}

export function softPlus() {
    let calc = (input) => {
        let activatedOuput = []

        //If the batch size is only 1
        if(input[0].length == undefined) {
            for(let i = 0; i < input.length; i++) {
                activatedOuput.push(Math.log(1 + Math.pow(Math.E, input[i])))
            }
        }
         //If there is an actual batch of samples
        else {
            for(let i = 0; i < input.length; i++) {
                let vector = []

                for(let j = 0; j < input[0].length; j++) {
                    vector.push(Math.log(1 + Math.pow(Math.E, input[i][j])))
                }
                activatedOuput.push(vector)
            }
            
        }
        return activatedOuput
    }

    let calcD = (x) => {
       return 1/(1 + Math.pow(Math.E, -x))
    }

    return { calc, calcD }
}

export function softMax() {
    let calc = (input) => {
        let activatedOuput = []

         //If the batch size is only 1
        if(input[0].length == undefined) {
            let maxInput = Math.max(...input)
            
            for(let i = 0; i < input.length; i++) {
              activatedOuput.push(Math.pow(Math.E, input[i] - maxInput))
            }

            const normalizerSum = activatedOuput.reduce((a,b) => a + b, 0) 
            for(let i = 0; i < input.length; i++) {
                activatedOuput[i]= activatedOuput[i] / normalizerSum
            }
        }
        //If there is an actual batch of samples
        else {
            let maxInputs =  input.map(vector => Math.max(...vector))
            for(let i = 0; i < input.length; i++) {
                let vector = []
                for(let j = 0; j < input[0].length; j++) {
                   vector.push(Math.pow(Math.E, input[i][j] - maxInputs[i]))
                }
                activatedOuput.push(vector)
            }
        
            for(let i = 0; i < input.length; i++) {
                const normalizerSum = activatedOuput[i].reduce((a,b) => a + b, 0) 
                for(let j = 0; j < input[0].length; j++) {
                    activatedOuput[i][j] /= normalizerSum
                }
            }
        }
        return activatedOuput
    }
    
    let calcD = (output, index) => {
        let derivative = 0

        for(let j = 0; j < output.length; j++) {
            let partialD

            if(index == j) {
                partialD = output[j] * (1 - output[j])
            } 
            else {
                partialD = - output[index] * output[j]
            }

            derivative += partialD
        }

        return  derivative
    }

    return { calc, calcD }
}

export function argMax() {
    let calc = (input) => {
        let activatedOuput = []

        //If the batch size is only 1
        if(input[0].length == undefined) {
            let maxInput = Math.max(...input)
            
            for(let i = 0; i < input.length; i++) {
                input[i] = input[i] == maxInput ? 1 : 0
            }
        }
        //If there is an actual batch of samples
        else {
            let maxInputs = input.map(vector => Math.max(...vector))
    
            for(let i = 0; i <input.length; i++) {
                let vector = []

                for(let j = 0; j <input[0].length; j++) {
                  vector.push(input[i][j] == maxInputs[i] ? 1 : 0)
                }
                activatedOuput.push(vector)
            }
        }
        return activatedOuput
    }

    let calcD = () => {
        return 0
    }

    return { calc, calcD }
}