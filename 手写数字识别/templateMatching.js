//使用模板匹配算法进行训练
const fs = require('fs')
var Jimp  = require("jimp") //裁剪操作图片像素的包

//将图片组转化为行向量的组合，每个行向量作为一张图片的特征
function img2ventor(path){
    let arr = []
    return new Promise((resolve, reject)=>{
        Jimp.read(path, (err, lenna)=>{
            lenna.resize(28, 28)
            if (err) throw err;
            let row  = lenna.bitmap.width
            let col = lenna.bitmap.height
            for(let i=0; i< row; i++){
                for(let j=0; j< col; j++){
                    arr.push(lenna.getPixelColor(i,j))
                }
            }
            resolve(arr)
        })
    })
}

//0-9数字模板的读取
function readTemplate(){
    let images = [] 

    for(let i=0; i< 9; i++){
        let sample = await img2ventor(`../手写/${i}/4.bmp`)
        images.push(sample)
    }
}

img2ventor('./1.bmp')