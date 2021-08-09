## Instructions on datasets preparation
You need to download the images of each dataset from their original sources. We recommend downloading the images to the `data/datasets/` directory. If you choose to download the images to a different directory, please make sure to update the path in the scripts accordingly. Please refer to the following instructions for more details.


### X-ray Datasets
* **MIMIC-CXR**: Download the images from [https://physionet.org/content/mimic-cxr-jpg/2.1.0/](https://physionet.org/content/mimic-cxr-jpg/2.1.0/). We also provided the metadata we organized in `data/datasets/MIMIC-CXR/MIMIC-CXR_metadata.json`, including the finds extracted from the radiology reports using GPT-4.

* **NIH-CXR**: Download the images from [https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data). An example image path in our split is `NIH-CXR/images_002/images/00001906_004.png`.

* **CheXpert**: Download the images from [https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2/](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2). An example image path in our split is `CheXpert-v1.0/train/patient10900/study25/view1_frontal.jpg`.

* **Pneumonia**: Download the images from [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). An example image path in our split is `pneumonia/images/test/PNEUMONIA/person112_bacteria_538.jpeg`.

* **COVID-QU**: Download the images from [https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). An example image path in our split is `COVID-QU/images/COVID-3330.png`.

* **Open-i**: Download the images from [https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university). An example image path in our split is `open-i/images/1907_IM-0589-1001.dcm.png`.

* **VinDr-CXR**: Download the images from [https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data). An example image path in our split is `vindr-cxr/images/e1eb9553f694d