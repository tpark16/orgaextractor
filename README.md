# OrgaExtractor
An easy-to-use deep learning-based image processing tool for organoid image analysis
<font size="1"> This repo is under review. </font>

![alt text](https://github.com/tpark16/orgaextractor/blob/main/OrgaExtractor_overview.png)


![alt text](https://github.com/tpark16/orgaextractor/blob/main/OrgaExtractor_confusionMatrix.png)
The number of organoids counted by OrgaExtractor. (a) Confusion matrix in the context of detection. (b) Confusion matrix in the context of instance segmentation.

OrgaExtractor can analysis ~82% of organoids in the image.

## Introduction
OrgaExtractor is designed to overcome the current inefficientcy in analyzing organoid images. It is Deep Learning based organoid segmentation algorithm that produces several metrics, which users can define themselves in need. OrgaExtractor was trained on 15 in-house colon organoid images and achieved dice similarity coefficient(DSC) of 0.867 for raw segmentation, 0.853 after post-processing. It also provides some basic metrics that reflect the actual organoid culture conditions, such as the number of organoids in the image, projected area of organoids, diameter, perimeter, major and minor axis length, eccentricity, roundness, solidity, and circularity. Users can freely define their metrics as long as they are calculable with [OpenCV](https://opencv.org/) libarary. 

OrgaExtractor is intended to be:
* User-friendly tool that requires minimal image adjustment for researchers who unfamiliar with programming. 
* tested and fine-tuned on user's custom dataset

OrgaExtractor does not require to install any dependencies to run. It is designed to run on Google Colab without any hardware limitation. Furthermode, we provide out-of-box method so that you can adapt the code in need.

## Using OrgaExtractor
You can use OrgaExtractor on either google colab or local computer.

### Running on Google Colab
If you are not familiar with programming, you can easily use OrgaExtractor by uploading your own dataset [here](https://colab.research.google.com/github/tpark16/orgaextractor/blob/main/Orgaextractor.ipynb)

It runs by executing all cells after you upload an image to colab's `test` folder. The result will be saved under `result` folder.


### Running on source code
To run OrgaExtractor in your local environment:
```
git clone https://github.com/tpark16/orgaextractor.git
cd orgaextractor
pip install -r requirements.txt
```
This copies orgaextractor repo into local computer and install required dependencies.

OrgaExtractor expects input image either numpy format(`.npy`) or image format(`.png`,`.jpeg` etc...) that OpenCV can afford. Once you complete formatting input data and create dataset path, you need to download our [pretrained model](https://drive.google.com/uc?id=1wOzvgroIgpEA9kaYfbz0Q3vUL5GY1my9) and set model path.

After setting all configurations, you can run python script:

```
python3 main.py --data_dir YOUR_DATASET_PATH --ckpt_dir DOWNLOADED_MODEL_PATH --result_dir DESIRABLE_RESULT_PATH --mode "test" 
```

You can also set `--fp16` to use mixed precision.

### Fine-tuning OrgaExtractor
To fine-tune OrgaExtractor with your own dataset:

```
python3 main.py --lr 1e-5 --batch_size 2 --num_epoch 100 --data_dir YOUR_DATASET_PATH --ckpt_dir DOWNLOADED_MODEL_PATH --result_dir DESIRABLE_RESULT_PATH --mode "train" --cuda_devices 0 --train_continue "on"
```

## Add custom metric to OrgaExtractor
You test your custom metric as long as the metric is calculable with OpenCV. Under `utils/postprocessing.py`, you will fine multiple metrics:

```python
def analysis(img_contour, contours, hie):
    ...

    ## Here you can add any metric you want for further anaylsis
    Eccentricity = round(np.sqrt(pow(a, 2) - pow(b, 2))/a, 2)
    perimeter = np.round(cv2.arcLength(x, True),2)
    circularity = (4*pi*area)/(perimeter**2)
    Roundness = (4*area) / (pi * (majorAxisLength**2))
    solidity = float(area) / hull_area
```

As you add the metric to test, you will find the result in exported excel file in the result folder.
For example, if you want the ratio of contour area to bounding rectangle area:

```python
_, _, rect_w, rect_h = cv2.boundingRect(x)
rect_area = rect_w * rect_h
extent = float(area)/rect_area
```

## Dataset
Our Dataset is available at [here](https://drive.google.com/drive/folders/17K4N7gEZUqAcwf9N2-I5DPbywwPvzAvo).

Dataset consists of three part; train, validation, and test. Each part has 15, 5, 10 organoid images, respectively

## Contact
Taeyun Park (ygj03084@yuhs.ac)
Taeyul K. Kim (kimkj1731@yuhs.ac)
