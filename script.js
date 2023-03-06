const uploadBox = document.querySelector(".upload-box");
const previewImg = uploadBox.querySelector("img");
const fileInput = uploadBox.querySelector("input");
const modal = document.querySelector('.modal')
const solutionContainerEl = document.getElementById("solution-container");

const classes = ['Cat', 'Dog']

async function prediction(inputTensor){
    showLoading();
    let featuresSession = await ort.InferenceSession.create("./feature_model.onnx");
    let feeds = { "input.1": inputTensor };
    let results = await featuresSession.run(feeds);
    
    let poolingSession = await ort.InferenceSession.create("./avgpool_model.onnx");
    feeds = { "input": results[165] };
    results = await poolingSession.run(feeds);
    
    let classifySession = await ort.InferenceSession.create("./classifier_model.onnx");
    const inputTensor3 = new ort.Tensor('float32', results[1].data, [1, 512*7*7]);
    
    feeds = { 'onnx::Gemm_0': inputTensor3 };
    results = await classifySession.run(feeds);

    resultData = results[12].data;
    let result = resultData.indexOf(Math.max(...resultData));
    solutionContainerEl.innerText = classes[result];
    hideLoading();
}

function exportToJsonFile(jsonData) {
    let dataStr = JSON.stringify(jsonData);
    let dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

    let exportFileDefaultName = 'data.json';

    let linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}

function processImage(canvas){
    var tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 224;
    tmpCanvas.height = 224;
    var tmpctx = tmpCanvas.getContext("2d");
    tmpctx.drawImage(canvas, 0,0,224,224);

    const imgData = tmpctx.getImageData(0, 0, 224, 224);
    const inputData = Float32Array.from(imgData.data)

    //exportToJsonFile({data:inputData})
    const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());
    let tmpData = []
    for (let i = 0; i< inputData.length; i+=4){
        redArray.push((inputData[i+0]/255 - 0.485)/0.229);
        greenArray.push((inputData[i+1]/255 - 0.456)/0.224);
        blueArray.push((inputData[i+2]/255 - 0.406)/0.225);
    }
    //Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
    const transposedData = redArray.concat(greenArray).concat(blueArray);
    
    const inputTensor = new ort.Tensor('float32', transposedData, [1, 3, 224, 224]);
    prediction(inputTensor)
}

const loadFile = (e)=>{
    const file = e.target.files[0];
    if (!file) return;
    previewImg.src = URL.createObjectURL(file);
    previewImg.addEventListener("load", ()=>{
        document.querySelector(".wrapper").classList.add("active");
        
        processImage(previewImg);
    })
}

fileInput.addEventListener("change", loadFile);

uploadBox.addEventListener("click", ()=>fileInput.click());

function showLoading(){
    modal.classList.add('open')
}
function hideLoading(){
    modal.classList.remove('open')
}