export function adaGrad(ep = 0.00000001) {
    const epsilon = ep

    let calc = (lRate, weights, biases, weightsD, biasesD, G, Gb) => {   
        let currentLRate, currentBLRate

        for(let i = 0; i < weightsD.length ; i++) {
            for(let j = 0; j < weightsD[0].length ; j++) {
                currentLRate = lRate / Math.sqrt(G[i][j] + epsilon)
                weights[i][j] -= weightsD[i][j] * currentLRate
            }

            currentBLRate = lRate / Math.sqrt(Gb[i] + epsilon)
            biases[i] -= biasesD[i] * currentBLRate 
        } 
    }

    let prep = (G, Gb, weightsD, biasesD) => {
        for(let i = 0; i < weightsD.length ; i++) {
            Gb[i] -=  Gb[i][j] + biasesD[i] * biasesD[i]

            for(let j = 0; j < weightsD[0].length ; j++) {
                 G[i][j] -=  G[i][j] + weightsD[i][j] * weightsD[i][j]
            }
        }
    }

    return { calc, prep }
}

export function adaDelta(y = 0.95) {
    const gamma = y

    let calc = (lRate, weights, biases, weightsD, biasesD, G, Gb) => { 
        let currentLRate, currentBLRate

        for(let i = 0; i < weightsD.length ; i++) {
            for(let j = 0; j < weightsD[0].length ; j++) {
                currentLRate = lRate / Math.sqrt(G[i][j] + epsilon)
                weights[i][j] -= weightsD[i][j] * currentLRate * m
            }

            currentBLRate = lRate / Math.sqrt(Gb[i] + epsilon)
            biases[i] -= biasesD[i] * currentBLRate 
        }    
    }

    let prep = (G, Gb, weightsD, biasesD) => {
        for(let i = 0; i < weightsD.length ; i++) {
            Gb[i] -=  gamma * Gb[i] + (1 - gamma) * biasesD[i] * biasesD[i]

            for(let j = 0; j < weightsD[0].length ; j++) {
                 G[i][j] -=   gamma * G[i][j] + (1 - gamma) * weightsD[i][j] * weightsD[i][j]
            }
        }
    }

    return { calc, prep }
}

export function momentum(momentum = 0.5) {
    const m = momentum 

    let calc = (lRate, weights, biases, weightsD, biasesD, G, Gb, change, changeB) => {    
        let newChange, newBChange

        for(let i = 0; i < weightsD.length ; i++) {
            for(let j = 0; j < weightsD[0].length ; j++) {
                newChange = lRate * weightsD[i][j] + m * change[i][j]
                change[i][j] = newChange
                weights[i][j] -= newChange
            }

            newBChange = lRate * biasesD[i] + m * changeB[i]
            biases[i] -=  newBChange
        } 
    }

    let prep = () => {
        return
    }

    return { calc, prep }
}

/*export function adam(b1, b2, ep = 0.00000001) {
    let beta1 = b1
    let beta2 = b2
    let epsilon = ep

    let calc = (lRate, weights, biases, weightsD, biasesD, G, Gb, change, changeB) => {
        let currentLRate, currentBLRate

        for(let i = 0; i < weightsD.length ; i++) {
            for(let j = 0; j < weightsD[0].length ; j++) {
                currentLRate = lRate / Math.sqrt(G[i][j] + epsilon)
                weights[i][j] -= weightsD[i][j] * currentLRate
            }

            currentBLRate = lRate / Math.sqrt(Gb[i] + epsilon)
            biases[i] -= biasesD[i] * currentBLRate 
        } 
    }
    }

    let prep = () => {

    }

    return { calc, prep }
}*/

export function decay(epochNum, lRate, decay) {
    return 1 / (1 + decay * epochNum) * lRate
}