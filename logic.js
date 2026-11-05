var onnxSess;
const imgElem = new Image();
var offsetX = 0,
  offsetY = 0;
var classes = [];
var classesBackup = [];

const DOM = {
  headerText: "header",
  modelFile: "modelFile",
  imageFile: "imageFile",
  cifarImage: "cifarImage",
  labelId: "labelId",
  labelsField: "labels",
  undoBtn: "undo",
  addBtn: "add",
  generateRandom: "random",
  resetBtn: "reset",
  nField: "N",
  predictedLabel: "prediction",
  imageShift: "shift",
  imageLeftShift: "left",
  imageUpShift: "up",
  imageRightShift: "right",
  imageDownShift: "down",
  histogramClasses: "histogram",
  controlsGroup: "settings"
};

const INSTRUCTION = {
  step1: "Step 1. Select trained ONNX model",
  step2: "Step 2. Upload an image for classification",
  step3: "Step 3. Select class labels and get predictions"
};

const mappingX = { left: -1, up: 0, right: 1, down: 0 };
const mappingY = { left: 0, up: -1, right: 0, down: 1 };

const find = (key) => {
  return document.getElementsByClassName(key)[0];
};

find(DOM.headerText).textContent = INSTRUCTION.step1;

document.onkeydown = checkKey;

function checkKey(e) {
  e = e || window.event;
  if (e.keyCode == "38") {
    e.preventDefault();
    offsetY -= 1;
    shiftImage();
    recognizeImage();
  } else if (e.keyCode == "40") {
    e.preventDefault();
    offsetY += 1;
    shiftImage();
    recognizeImage();
  } else if (e.keyCode == "37") {
    e.preventDefault();
    offsetX -= 1;
    shiftImage();
    recognizeImage();
  } else if (e.keyCode == "39") {
    e.preventDefault();
    offsetX += 1;
    shiftImage();
    recognizeImage();
  }
}

const shiftImage = () => {
  const img = find(DOM.cifarImage);
  let canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext("2d");

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, 32, 32);
  ctx.fillStyle = "black";
  ctx.fill();
  ctx.restore();

  ctx.drawImage(imgElem, offsetX, offsetY, 32, 32);
  img.src = canvas.toDataURL();
};

Array.from(document.getElementsByClassName(DOM.imageShift)).map((elem) => {
  elem.addEventListener("click", (e) => {
    const action = e.srcElement.classList[1];
    offsetX += mappingX[action];
    offsetY += mappingY[action];
    shiftImage();
    recognizeImage();
  });
});

const drawHist = (preds) => {
  const canvas = find(DOM.histogramClasses);
  const octx = canvas.getContext("2d");

  octx.width = canvas.width;
  octx.height = canvas.height;
  octx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0, nums = 5; i <= nums; i++) {
    octx.save();
    octx.shadowColor = "rgba(0,0,0,0.25)";
    octx.moveTo(0.5 + 0, (128 * (nums - i)) / nums);
    octx.lineTo(0.5 + canvas.width, (128 * (nums - i)) / nums);
    octx.font = "12px serif";
    octx.fillText(
      `${Math.round((i / nums) * 100)}`,
      canvas.width - 15,
      (128 * (nums - i)) / nums - 5
    );
    octx.fillText(
      `${Math.round((i / nums) * 100)}`,
      0,
      (128 * (nums - i)) / nums - 5
    );

    octx.stroke();
    octx.restore();
  }

  const step = Math.floor((octx.width / preds.length) * 0.9);
  const offset = Math.floor(octx.width * 0.05);
  octx.fillStyle = "black";

  for (let i = 0; i < preds.length; i++) {
    const barHeight = Math.floor((preds[i] * octx.height) / 2);
    octx.save();
    octx.rotate(-Math.PI / 2);
    octx.textAlign = "right";
    octx.font = "20px serif";
    octx.fillText(
      mapping[classes[i]],
      -octx.height / 2,
      offset + step * (i + 0.5)
    );
    octx.restore();

    octx.save();
    octx.beginPath();
    octx.rect(offset + step * i, octx.height / 2 - barHeight, step, barHeight);
    octx.fillStyle = "white";
    octx.fill();
    octx.restore();

    octx.save();
    octx.beginPath();
    octx.strokeRect(
      offset + step * i,
      octx.height / 2 - barHeight,
      step,
      barHeight
    );
    octx.fillStyle = "black";
    octx.fill();
    octx.restore();
  }
};

find(DOM.modelFile).onchange = async function (event) {
  var fileList = this.files;
  if (!fileList.length) {
    find(DOM.imageFile).style.display = "none";
    find(DOM.imageFile).value = "";
    find(DOM.cifarImage).parentElement.style.display = "none";
    find(DOM.controlsGroup).style.display = "none";
    find(DOM.resetBtn).click();
    find(DOM.headerText).textContent = INSTRUCTION.step1;
    return;
  }
  let file = fileList[0];
  let reader = new FileReader();
  reader.onloadend = async function () {
    try {
      onnxSess = await ort.InferenceSession.create(reader.result);
      
      const img = new Float32Array(32 * 32 * 3);
      img.fill(1);
      const input = new ort.Tensor('float32', img, [1, 32, 32, 3]);
      
      const inputName = onnxSess.inputNames[0];
      const feeds = {};
      feeds[inputName] = input;
      
      await onnxSess.run(feeds);
      
      find(DOM.imageFile).style.display = "block";
      find(DOM.headerText).textContent = INSTRUCTION.step2;
    } catch (e) {
      console.error("Ошибка загрузки модели:", e);
    }
  };
  reader.readAsArrayBuffer(file);
};

const applySoftmax = (logits) => {
  const maxLogit = Math.max(...logits);
  const scores = logits.map((l) => Math.exp(l - maxLogit));
  const denom = scores.reduce((a, b) => a + b);
  return scores.map((s) => s / denom);
};

const recognizeImage = async () => {
  const imgVis = find(DOM.cifarImage);
  if (!imgVis.complete) {
    await new Promise((r) => {
      imgVis.onload = r;
    }).then();
  }
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgVis, 0, 0, 32, 32);
  const imgArray = Array.from(ctx.getImageData(0, 0, 32, 32).data).filter(
    (e, i) => (i + 1) % 4
  );
  canvas.remove();

  try {
    const imgFloat32 = Float32Array.from(imgArray);
    const input = new ort.Tensor('float32', imgFloat32, [1, 32, 32, 3]);
    
    const inputName = onnxSess.inputNames[0];
    const feeds = {};
    feeds[inputName] = input;
    
    const results = await onnxSess.run(feeds);
    const outputName = onnxSess.outputNames[0];
    const output = results[outputName].data;
    const output_slice = [...output];
    
    // Блокируем предсказание и отрисовку, пока не будут добавлены все 3 класса
    if (classes.length !== 3) {
      find(DOM.predictedLabel).textContent = "Вводите по очереди 12, 42, 75 и жмите Add";
      return;
    }
    
    find(DOM.predictedLabel).textContent = mapping[
      classes[indexOfMax(output_slice)]
    ].toUpperCase();
    drawHist(applySoftmax(output_slice));
  } catch (e) {
    console.error("Ошибка при распознавании картинки:", e);
  }
};

find(DOM.imageFile).onchange = function (event) {
  var fileList = this.files;
  const imgVis = find(DOM.cifarImage);
  const controlsGroup = find(DOM.controlsGroup);
  if (!fileList.length) {
    imgVis.parentElement.style.display = "none";
    controlsGroup.style.display = "none";
    find(DOM.headerText).textContent = INSTRUCTION.step2;
    return;
  }
  offsetX = 0;
  offsetY = 0;
  let file = fileList[0];
  let reader = new FileReader();
  reader.onloadend = async function () {
    imgElem.src = reader.result;
    if (!imgElem.complete) {
      await new Promise((r) => {
        imgElem.onload = r;
      }).then();
    }
    imgVis.src = imgElem.src;
    recognizeImage();
  };
  reader.readAsDataURL(file);
  imgVis.parentElement.style.display = "block";
  controlsGroup.style.display = "block";
  find(DOM.headerText).textContent = INSTRUCTION.step3;
};

function indexOfMax(arr) {
  if (arr.length === 0) {
    return -1;
  }
  var max = arr[0];
  var maxIndex = 0;
  for (var i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }
  return maxIndex;
}

find(DOM.labelId).addEventListener("change", function () {
  let v = parseInt(this.value);
  if (v < 0) this.value = 0;
  if (v > 99) this.value = 99;
});

const appendLabelId = (labelId, backup = true) => {
  if (classes.indexOf(labelId) >= 0) {
    return;
  }
  if (backup) {
    classesBackup.push([...classes]);
  }
  classes.push(labelId);
  classes.sort((a, b) => a - b);
};

find(DOM.addBtn).addEventListener("click", (e) => {
  const labelId = find(DOM.labelId);
  if (!labelId.value.length) {
    return;
  }
  const value = parseInt(labelId.value);
  appendLabelId(value);
  find(DOM.nField).textContent = classes;
  recognizeImage();
});

find(DOM.generateRandom).addEventListener("click", (e) => {
  classesBackup.push([...classes]);
  classes.length = 0;
  for (let i = 0; (i < 100) & (classes.length < 10); i++) {
    const label = Math.floor(Math.random() * 100);
    appendLabelId(label, false);
  }
  find(DOM.nField).textContent = classes;
  recognizeImage();
});

find(DOM.undoBtn).addEventListener("click", (e) => {
  if (!classesBackup.length) return;
  classes = classesBackup.pop();
  find(DOM.nField).textContent = classes;
  recognizeImage();
});

find(DOM.resetBtn).addEventListener("click", (e) => {
  classes.length = 0;
  classesBackup.length = 0;
  find(DOM.nField).textContent = classes;
});

const mapping = [
  "apple",
  "aquarium_fish",
  "baby",
  "bear",
  "beaver",
  "bed",
  "bee",
  "beetle",
  "bicycle",
  "bottle",
  "bowl",
  "boy",
  "bridge",
  "bus",
  "butterfly",
  "camel",
  "can",
  "castle",
  "caterpillar",
  "cattle",
  "chair",
  "chimpanzee",
  "clock",
  "cloud",
  "cockroach",
  "couch",
  "crab",
  "crocodile",
  "cup",
  "dinosaur",
  "dolphin",
  "elephant",
  "flatfish",
  "forest",
  "fox",
  "girl",
  "hamster",
  "house",
  "kangaroo",
  "keyboard",
  "lamp",
  "lawn_mower",
  "leopard",
  "lion",
  "lizard",
  "lobster",
  "man",
  "maple_tree",
  "motorcycle",
  "mountain",
  "mouse",
  "mushroom",
  "oak_tree",
  "orange",
  "orchid",
  "otter",
  "palm_tree",
  "pear",
  "pickup_truck",
  "pine_tree",
  "plain",
  "plate",
  "poppy",
  "porcupine",
  "possum",
  "rabbit",
  "raccoon",
  "ray",
  "road",
  "rocket",
  "rose",
  "sea",
  "seal",
  "shark",
  "shrew",
  "skunk",
  "skyscraper",
  "snail",
  "snake",
  "spider",
  "squirrel",
  "streetcar",
  "sunflower",
  "sweet_pepper",
  "table",
  "tank",
  "telephone",
  "television",
  "tiger",
  "tractor",
  "train",
  "trout",
  "tulip",
  "turtle",
  "wardrobe",
  "whale",
  "willow_tree",
  "wolf",
  "woman",
  "worm"
];