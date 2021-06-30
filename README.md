# Object Detection And Image Recognition (Batch Processing)

## Goal
The goal of this project is to train the computer to recognize objects on an image. 

## Inspiration & Use Cases
- Auto collecting foot traffic information
Foot traffic is an important indicator in determining business location. To measure foot traffic, a camera could be set and upload images of the street hourly. Once we get the image, a computer can read and count how many people are on the street at a given time. 

- Auto labeling amenities in an apartment/hotel/airbnb/house listing
When listing an apartment on a marketplace, it could be tedious to manually fill out all the amenities available. To auto recognize amenities, we could utilize the existing photos to auto-tag and fill out. 

## Demo 1: Recognize kitchen amenities

The model will automatically recognize kitchen amenities and then export them to a csv file.
![kitchen](https://user-images.githubusercontent.com/44503223/123950427-48bbe900-d969-11eb-8e60-8918c7db7e04.gif)

## Demo 2: Count number of people



## Detailed Process

To start, we will load pre-trained yolo model. 

`#load yolo
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))`


## Learn More

You can learn more in [Tingting Duan's Project Portfolio](https://tingting0618.github.io).

