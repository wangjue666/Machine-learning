const cv = require('opencv4nodejs');
const classifier = new cv.CascadeClassifier("./haarcascade_frontalface_alt.xml");
async function run() {
    try {
        const img = await cv.imreadAsync('./imgs/face.png');
        const grayImg = await img.bgrToGrayAsync();
        const { objects , numDetections } = await classifier.detectMultiScaleAsync(grayImg);
        console.log(objects, numDetections)
        cv.drawDetection(
            img,
            objects[0],
            { color: new cv.Vec(255, 0, 0), segmentFraction: 4 }
        );
        cv.imwriteAsync('./dist.png', img,(err) => {})
    } catch (err) {
        console.error(err);
    }
}
run()