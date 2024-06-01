export function scaleByMatrix(matrix1, matrix2) {
    if(matrix1.length !== matrix2.length || matrix1[0].length !== matrix2[0].length ) {
        return `shape error: can't scale matrix by matrix, ${matrix1.length} * ${matrix1[0].length} != ${matrix2.length} * ${matrix2[0].length}`
    }
    let output = []

    for(let i = 0; i < matrix1.length; i++) {
        let vector = []
        for(let j = 0; j < matrix1[0].length; j++) {
            vector[j] = matrix1[i][j] * matrix2[i][j]
        }
        output.push(vector)
    }
    return output
}

export function dotProduct(vector1, vector2) {
    if(vector1.length !== vector2.length) {
        return `shape error: can't do the dot product on the given vectors, ${vector1.length} != ${vector2.length}`
    }
    let output = 0

    for(let i = 0; i < vector1.length; i++) {
      output += vector1[i] * vector2[i]
    }
    return output
}

export function transpose(matrix) {
    let [row] = matrix
    return row.map((value, column) => matrix.map(row => row[column]))
}

export function matrixMul(matrix1, matrix2) {
    const i_m1 = matrix1[0].length   
    const j_m1 = matrix1.length;  
    const i_m2 = matrix2[0].length
    const j_m2 = matrix2.length; 

    const isVector = matrix1[0].constructor !== Array 
    const isInvalidVector= isVector && j_m1 != j_m2
   
    if (isInvalidVector && (i_m1 != j_m2)) return `shape error: given matrixes can't be multiplied`

    let multiplication = new Array(j_m1);  
    for (let x=0; x<multiplication.length;x++) {
        multiplication[x] = new Array(i_m2).fill(0);
    }      

    for (let x=0; x < multiplication.length; x++) {      
        for (let y=0; y < multiplication[x].length; y++) {   
            for (let z=0; z<i_m1; z++) {              
                       multiplication [x][y] = multiplication [x][y] + matrix1[x][z]*matrix2[z][y]; 
            }      
        }  
    }
    return multiplication
}


function calcOutput(input) {
    if(this.weights[0].length !== input.length || input.length !== this.biases.length) {
        return `shape error: can't calculate the output, weights = (${this.weights.length},${this.weights[0].length})
        input = (${input.length}), biases = (${this.biases.length})`
    }
    let output = []

    for(let i = 0; i < weights.length; i++) {
       const result = math.dotProduct(this.weights[i], input) + this.biases[i]
       output.push(result)
    }
    this.output = output
}

