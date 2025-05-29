# Smart-Prosthodontics


---

## 📌 Overview

**Smart-Prosthodontics** is an AI-powered dental imaging tool that performs:

* Automated **tooth detection** using YOLOv11
* **Inpainting of missing teeth** with Stable Diffusion fine-tuned for medical X-rays
* Optional **3D reconstruction** using statistical shape models

Built with **Python**, **PyTorch**, **OpenCV**, and GPU acceleration, DeepTeeth addresses critical diagnostic challenges in dentistry by combining robust object detection with powerful generative modeling.

---

## 🛠️ Features

* 🧠 **YOLOv11-based detection** for high-speed and accurate tooth localization
* 🦷 **Mask-based inpainting** with Stable Diffusion (grayscale dental domain)
* 📐 **Spatial analysis**: bounding boxes, tooth gap measurement, and distance calculations
* 🖼️ **Realistic X-ray reconstruction** using custom prompts and diffusion
* 📊 **Model training and evaluation** using mAP, loss curves, and visual outputs

---

## 🔁 Workflow

```
1. Upload raw dental X-rays
2. Annotate and preprocess images
3. Train YOLOv11 for tooth detection
4. Detect missing teeth and generate masks
5. Use Stable Diffusion to inpaint missing teeth
6. Produce output-ready diagnostic X-rays
```


---

## 🚀 Technologies Used

* 🧠 **YOLOv11**: Custom trained for medical imaging
* 🎨 **Stable Diffusion 2.1**: Inpainting model for grayscale dental images
* 🔥 **PyTorch**: Deep learning framework
* 🖼️ **Roboflow**: Dataset annotation & preprocessing
* 🖥️ **OpenCV**: Image handling and visualization

---

## 🧪 Sample Inpainting Code

```python
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

original_image = Image.open("path_to_xray.jpg").convert("RGB")
mask_image = Image.open("path_to_mask.png").convert("L")

result = pipe(
    prompt="realistic upper premolar tooth with roots, grayscale dental x-ray, high detail",
    negative_prompt="blurry, distorted, lowres",
    image=original_image,
    mask_image=mask_image,
    strength=0.9,
    guidance_scale=8.0,
    num_inference_steps=50,
).images[0]

result.save("inpainted_xray.png")
```

---

## 📈 Model Performance

* 📦 **Box Loss** reduced from `1.50` to `1.15`
* 🧪 **Class Loss** improved from `2.5` to `0.5`
* 🔁 **Cisco Loss** decreased consistently
* ✅ **Stable convergence** across training epochs

---

## 🖼️ Outputs

1. Original X-ray Image

 ![495](https://github.com/user-attachments/assets/3c78d17a-5e58-4030-bc16-8a3d86a0742b)
 
2. YOLOv11 Detection
   
    ![yolo_identified](https://github.com/user-attachments/assets/dc9b4e7a-ed4a-46cb-a7d0-3bccf9e5aec2)

3. Tooth Mask Region
   
   ![missing_tooth_14_mask](https://github.com/user-attachments/assets/c477b102-5a3b-4b77-8daf-751ab34a4d50)

4. Stable Diffusion Inpainted Result

    ![inpainted_xray](https://github.com/user-attachments/assets/5f10886e-df5b-4dba-9be0-c8180fa7fef8)

---


## 👨‍⚕️ Applications

* 🦷 Dental treatment planning
* 📚 Educational simulation
* 🧬 Biomedical imaging research
* 🔬 Pre-operative diagnosis and patient reports

---

## 📄 Project Info

* 🎓 **Institution**: Coimbatore Institute of Technology
* 🧑‍💻 **Author**: Surya S (71762134050)
* 📅 **Batch**: 2021–2026 

---

## 📎 License

This project is for academic and research purposes only. Please consult the author before commercial use.

---

Would you like me to export this as a `README.md` file or include image placeholders for visuals?
