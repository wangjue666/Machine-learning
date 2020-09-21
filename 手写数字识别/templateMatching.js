//使用模板匹配算法进行训练
var Jimp  = require("jimp") //裁剪操作图片像素的包

let correct_num = 0
let images = []
async function run(){
    images = await readTemplate()
    await testMatch('手写数字')
    console.log(`手写共测试10个样本，正确匹配个数为${correct_num}个`)
    correct_num = 0
    await testMatch('车牌')
    console.log(`车牌共测试10个样本，正确匹配个数为${correct_num}个`)
}
run()
//将图片组转化为行向量的组合，每个行向量作为一张图片的特征
function img2ventor(path){
    let arr = []
    return new Promise((resolve, reject)=>{
        Jimp.read(path, (err, lenna)=>{
            if(err){
                console.log(err)
            }
            lenna.resize(28, 28)
            if (err) throw err;
            let row  = lenna.bitmap.width
            for(let i=0; i< row; i++){
                arr[i] = []
            }
            lenna.scan(0, 0, lenna.bitmap.width, lenna.bitmap.height, function(x, y, idx) {
                var red = this.bitmap.data[idx + 0];
                var green = this.bitmap.data[idx + 1];
                var blue = this.bitmap.data[idx + 2];
                arr[x][y]= red 
            });
            resolve(arr)
        })
    })
}
//0-9数字模板的读取
async function readTemplate(){
    let images = [] 

    for(let i=0; i<= 9; i++){
        let sample = await img2ventor(`./手写数字/${i}/2.bmp`)
        images.push(sample)
    }

    return images
}

//测试匹配样本
async function testMatch(name){
    for(let i=0; i<= 9; i++){
        let distance = []
        let fileName 
        if(name == '手写数字'){
            fileName = `./${name}/${i}/4.bmp`  
        }else{
            fileName = `./${name}/${i}.1.bmp`   
        }
       
        let sample = await img2ventor(fileName)
        images.forEach(item=>{
            distance.push(compEuclidean(sample, item))
        })
        let minDis = Math.min(...distance)
        let minIdx = distance.indexOf(minDis)
        if (minIdx == i){
            correct_num++;
        }
        console.log(
            `数字${i}到模板的最小距离为：${minDis},匹配到的类别为：${minIdx}\n`
        )
    }
}

//计算俩像素的欧拉距离
function compEuclidean(p1, p2){
    let subs = []
    for(let i=0;i<p1.length;i++){
        for(let j=0; j<p1[0].length; j++){
            subs.push(Math.pow(p1[i][j]-p2[i][j], 2))
        }
    }
    let sums = subs.reduce((pre, cur)=>{
        return pre+cur
    })
    
    return Math.sqrt(sums, 2)
}

