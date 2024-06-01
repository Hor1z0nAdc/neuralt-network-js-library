export function accuracyClass(outputs, class_targets) {
    let accuracyArray = []

    const isSingleSample = outputs[0].length == undefined
    const isSingleOutput = (isSingleSample && outputs.length == 1) || outputs[0].length == 1
    const isTargetSingleFormated = class_targets[0].length == undefined
    // const isTargetSingleFormated =  (isSingleSample && !isSingleOutput && class_targets.length == 1) ||
    //                                 (!isSingleSample && !isSingleOutput && class_targets[0].length == undefined ) ||
    //                                 isSingleOutput

    //If the batch size is only 1
    if(isSingleSample) {
        //Only one output neuron
        if(isSingleOutput) {
            if((outputs > 0.5 && class_targets[0] == 1) || (outputs < 0.5 && class_targets[0] == 0)) accuracyArray.push(1)
            else accuracyArray.push(0)
        }
        else {
            //Determine the largest output of all the output neurons
            const index = outputs.indexOf(Math.max(...outputs))

            //If the class targets are formated "[1]" - only contains the largest output
            if(isTargetSingleFormated) {
                if(index == class_targets[0]) accuracyArray.push(1)
                else accuracyArray.push(0)

            }

            //If the class targets are formated "[0,1,0]" - contains target value for all the neurons
            else {
                const targetIndex = class_targets[0].indexOf(1)
                if(index == targetIndex) accuracyArray.push(1)
                else accuracyArray.push(0)
            }

        }
    }

    //If the output is a batch of samples
    else {
        //Only one output neuron
        if(isSingleOutput) {
            for(let i = 0; i < outputs.length; i++) {
                if(outputs[i][0] > 0.5) accuracyArray.push(1)
                else accuracyArray.push(0)
            }
        }
        else {
           //If the class targets are formated "[1]" - only contains the largest output
           if(isTargetSingleFormated) {
               for(let i = 0; i < outputs.length; i++) {

                   //Determine the largest output of all the output neurons
                   const vector = outputs[i]
                   const index = vector.indexOf(Math.max(...vector))
   
                   if(index == class_targets[i]) accuracyArray.push(1)
                   else accuracyArray.push(0)
               }
           }

           //If the class targets are formated "[0,1,0]" - contains target value for all the neurons
           else {
               for(let i = 0; i < outputs.length; i++) {
                   //Determine the largest output of all the output neurons
                   const vector = outputs[i]
                   const index = vector.indexOf(Math.max(...vector))
                   const targetIndex = class_targets[i].indexOf(1)

                    if(index == targetIndex) accuracyArray.push(1)
                    else accuracyArray.push(0)
                }
            }
        }
    }

     //Return the average accuracy on the random batch
     return accuracyArray.reduce((a, b) => a + b, 0) / accuracyArray.length
}

export function categoricalCrossEntropy() {
    let calc = (outputs, class_targets) => {
        let losses = []
        let targeted_output

        //If the batch size is only 1
        if(outputs[0].length == undefined) {
            if(class_targets[0].length == undefined) {
                targeted_output = outputs[class_targets] == 0 ? 100 : outputs[class_targets]
                losses.push(-Math.log(targeted_output))
            }
            else {
                let index = class_targets[0].indexOf(1)
                targeted_output = outputs[index] == 0 ? 100 : outputs[index]
                losses.push(-Math.log(targeted_output))
            }
        }
        //If there is an actual batch of samples
        else {
            if(class_targets[0].length == undefined) {
                for(let i = 0; i < outputs.length; i++) {
                    //Taking care of zero value, that would give back infinity
                    targeted_output = outputs[i][class_targets[i]] == 0 ? 100 : outputs[i][class_targets[i]]
                    losses.push(-Math.log(targeted_output))
                }
            }
            else {
                for(let i = 0; i < outputs.length; i++) {
                    let index = class_targets[i].indexOf(1)
                    targeted_output = outputs[i][index] == 0 ? 100 : outputs[i][index]
                    losses.push(-Math.log(targeted_output))
                }
            }
        }
        
        //Return the average of losses calculated on the random batch 
        let result = losses.reduce((a, b) => a + b, 0) / losses.length;
         return result
      
    }

    let calcD = (output, target, index) => {
        let costD = 0
    
       //if target determines the expected highest ouptut value
       if(target[0].length == undefined) {
        
           //calculate the derivative of every sample in the batch
           for(let i = 0; i < output.length; i++) {
               let y = index == target[i] ? 1 : 0
               costD += output[i][index] - y
            }
       }
       //if target is given in one-hot encodyng format
       else {   

            //calculate the derivative of every sample in the batch
            for(let i = 0; i < output.length; i++) {
                costD += output[i][index] - target[i][index]
            }
       }

       //calc and return the avarage of cost derivatives 
       return costD / output.length
    }

    return {calc, calcD}
}

export function binaryCrossEntropy() {
    let calc = (outputs, class_targets) => {

         //If the batch size is only 1
         if(outputs[0].length == undefined) {
            let res 
            let input = outputs[0]

            if(class_targets[0] == 1) {
                input = input == 0 ? 0.001 : input
                res =  - Math.log(input)
                res = res == -0? 0 : res
            }
            else {
                input = input == 1 ? 0.999 : input
                res = - Math.log(1 - input)
                res = res == -0? 0 : res
            }
            return res
         }
         //If there is an actual batch of samples
         else {
            let losses = []
            let loss, input

            for(let i = 0; i < outputs.length; i++) {
                input = outputs[i][0]
                
                if(class_targets[i] == 1) {
                    input = input == 0 ? 0.001 : input
                    loss = - Math.log(input)
                }
                else {
                    input = input == 1 ? 0.999 : input
                    loss = - Math.log(1 - input)
                }
                loss = loss == -0 ? 0 : loss
                losses.push(loss)
            }
            
            let result = losses.reduce((a, b) => a + b, 0) / losses.length;
            if(isFinite(result)) return result
            else return 9007199254740
         }
    }
    
    let calcD = (output, target) => {
        let costD = 0
        let firstT, secondT
        
        //for every sample in the batch calc. the derivative and add it to the common variable
        for(let i = 0; i < target.length; i++) {
            firstT = -(target[i] / output[i][0])
            secondT = (1 - target[i]) / (1 - output[i][0])
            costD += firstT + secondT
        }

        //return the average derivative of the batch
        return costD / target.length
    }

    return { calc, calcD }
}

export function meanError(outputs, targets) {
    let diff
    let loss = 0
  
    //If the batch size is only 1
    if(outputs[0].length == undefined) {
        for(let i = 0; i < outputs.length; i++) {
            diff = targets[i] - outputs[i] 
            loss += diff
        }
        loss /= outputs.length
        return loss
    }   
    //If there is an actual batch of samples
    else {
        let losses = []

        for(let i = 0; i < outputs.length; i++) {
            loss = 0

            //if the target values are given in this format: [10,20,30]
            if(targets[0].length == undefined) {
                  diff = targets[i] - outputs[i]
                  loss += diff
                  losses.push(loss)
            }
             //if the target values are given in this format: [[10],[20],[30]]
             //in case of multiple output neurons the target values have to be in that format
            else {
                for(let j = 0; j < outputs[0].length; j++) {
                  diff = targets[i][j] - outputs[i][j]
                  loss += diff
                }
                loss /= outputs[i].length
                losses.push(loss)
            }
        }

        let result = losses.reduce((a, b) => a + b, 0) / losses.length
        result == -0 ? 0 : result

        if(isFinite(result)) return result
        else return 9007199254740
    }
}

export function meanSquaredError() {
    let calc = (outputs, targets) => {
        let diff
        let loss = 0
      
        //If the batch size is only 1
        if(outputs[0].length == undefined) {
            for(let i = 0; i < outputs.length; i++) {
                diff = targets[i] - outputs[i] 
                loss += Math.pow(diff, 2)
            }
            loss /= outputs.length
            return loss
        }   
        //If there is an actual batch of samples
        else {
            let losses = []

            for(let i = 0; i < outputs.length; i++) {
                loss = 0

                //if the target values are given in this format: [10,20,30]
                if(targets[0].length == undefined) {
                      diff = targets[i] - outputs[i]
                      loss += Math.pow(diff, 2)
                      losses.push(loss)
                }
                 //if the target values are given in this format: [[10],[20],[30]]
                 //in case of multiple output neurons the target values have to be in that format
                else {
                    for(let j = 0; j < outputs[0].length; j++) {
                      diff = targets[i][j] - outputs[i][j]
                      loss += Math.pow(diff, 2)
                    }
                    loss /= outputs[i].length
                    losses.push(loss)
                }
            }

            let result = losses.reduce((a, b) => a + b, 0) / losses.length
            result == -0 ? 0 : result

            if(isFinite(result)) return result
            else return 9007199254740
        }
    }

    let calcD = (b, expected) => {
        let result = -2 * (expected - b)
        result == -0 ? 0 : result
        if(isFinite(result)) return result
        else return 10000
    }

    let calcDNumerical = (b, difference) => {
        return (calc(b + difference) - calc(b)) / difference
    }

    return { calc, calcD, calcDNumerical}
}

export function meanAbsoluteError() {
    let calc = (outputs, targets) => {
        let diff
        let loss = 0
      
        //If the batch size is only 1
        if(outputs[0].length == undefined) {
            for(let i = 0; i < outputs.length; i++) {
                diff = targets[i] - outputs[i] 
                loss += Math.abs(diff)
            }
            loss /= outputs.length
            return loss
        }   
        //If there is an actual batch of samples
        else {
            let losses = []

            for(let i = 0; i < outputs.length; i++) {
                loss = 0

                //if the target values are given in this format: [10,20,30]
                if(targets[0].length == undefined) {
                      diff = targets[i] - outputs[i]
                      loss += Math.abs(diff)
                      losses.push(loss)
                }
                 //if the target values are given in this format: [[10],[20],[30]]
                 //in case of multiple output neurons the target values have to be in that format
                else {
                    for(let j = 0; j < outputs[0].length; j++) {
                      diff = targets[i][j] - outputs[i][j]
                      loss += Math.abs(diff)
                    }
                    loss /= outputs[i].length
                    losses.push(loss)
                }
            }

            let result = losses.reduce((a, b) => a + b, 0) / losses.length
            return result == -0 ? 0 : result
        }
    }

    let calcD = (b, expected) => {
        if(expected > b) return -1
        else return 1
    }

    return { calc, calcD }
}
