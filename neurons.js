const datas = [
    [[0,0], 0],
    [[0,1], 1],
    [[1,0], 1],
    [[1,1], 1],
]

const weights = []
for(let i=0;i<3;i++){
    weights[i] = Math.random()-0.5
}

//激活
function calcOutput(inputs){
    let output = inputs[0]*weights[0] + inputs[1]*weights[1] + 1*weights[2]
    return sigmoid(output)
}




function sigmoid(x){
    return 1 / (1+Math.pow(Math.E, -x))
}

//计算误差大小
function errRate(output, excepted){
    return Math.abs(output - excepted)
}

//计算最近N次的错误率的平均值
const errors = []
const maxError = 20
function calcError(err){
    errors.push(err)
    if(errors.length > maxError){
        errors.shift()
    }
    return errors.reduce((tem,item)=>tem+item) / errors.length
}
//试着调整每个权重
const d = 0.000001
const trainRate = 0.01
const threshold = 0.001
function train(inputs, excepted){
    let err = errRate(calcOutput(inputs), excepted)

    const dw = []
    weights.forEach((w, i)=>{
        weights[i]+=d
        let err2 = errRate(calcOutput(inputs), excepted)
        dw[i] = (err2 - err) / d
        weights[i] = w
    })
    weights.forEach((w, i)=>{
        weights[i] -= dw[i] * trainRate
    })
    //调整每个权重 +0.00001
    //再次计算新的错误率

    let e = calcError(err)
    return e <= threshold
}


for(let i=0;;i++){
    let data = datas[i%datas.length]
    if(train(data[0], data[1])){
        break;
    }
}
console.log(weights)