from django.shortcuts import render  
from django.core.files.storage import FileSystemStorage  
import onnxruntime  
import numpy as np  
from PIL import Image  
from io import BytesIO  
import base64  
from torchvision import transforms   
import os
from django.conf import settings
  
# Шаг 4.1: Ограничение на 3 класса по твоему варианту (17, 47, 77)
imageClassList = {17: 'castle', 47: 'maple_tree', 77: 'snail'} #
  
def scoreImagePage(request):  
    return render(request, 'scorepage.html')  
  
def predictImage(request):  
    fileObj = request.FILES['filePath']  
    fs = FileSystemStorage()  
    filePathName = fs.save('images/'+fileObj.name,fileObj)  
    filePathName = fs.url(filePathName)  
    modelName = request.POST.get('modelName')  
    scorePrediction, img_uri = predictImageData(modelName, '.'+filePathName)  
    context = {'scorePrediction': scorePrediction, 'filePathName': filePathName, 'img_uri': img_uri}  
    return render(request, 'scorepage.html', context)  
  
def predictImageData(modelName, filePath):  
    img = Image.open(filePath).convert("RGB")  
    resized_img = img.resize((32, 32), Image.Resampling.LANCZOS)  
    img_uri = to_data_uri(resized_img)  
    
    input_image = Image.open(filePath).convert("RGB")  
    preprocess = transforms.Compose([  
        transforms.Resize(32),  
        transforms.CenterCrop(32),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])  
    input_tensor = preprocess(input_image)  
    input_batch = input_tensor.unsqueeze(0)  
     
    # Путь к вашей модели MobileNetV2
    model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'cifar100_mobilenetv2.onnx')
    sess = onnxruntime.InferenceSession(model_path) 
    
    # Получаем сырые вероятности для всех 100 классов
    raw_outputs = sess.run(None, {'input': to_numpy(input_batch)})[0][0]  
    
    # Ограничиваем выбор строго тремя классами по варианту
    target_classes = [17, 47, 77]
    target_logits = [raw_outputs[i] for i in target_classes]
    
    # Находим победителя (индекс 0, 1 или 2) и сопоставляем с реальным классом
    best_idx = np.argmax(target_logits)
    outputOFModel = target_classes[best_idx]
    
    # Забираем название из словаря imageClassList
    score = imageClassList[outputOFModel]  
  
    return score, img_uri
  
def to_numpy(tensor):  
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()  
  
def to_image(numpy_img):  
    img = Image.fromarray(numpy_img, 'RGB')  
    return img  
  
def to_data_uri(pil_img):  
    data = BytesIO()  
    pil_img.save(data, "JPEG") 
    data64 = base64.b64encode(data.getvalue())  
    return u'data:image/jpeg;base64,' + data64.decode('utf-8')