import requests
from PIL import Image
import torch
import sys
import os
from io import BytesIO
from transformers import AutoProcessor, Owlv2ForObjectDetection


# segment-anything model
# from segment_anything import sam_model_registry, SamPredictor
# import torch
# import cv2

#print(torch.cuda.is_available())

def is_url(path):
    return path.startswith("http://") or path.startswith("https://")


def owlv2_results(image_source, query):
# Load image based on source type
    if is_url(image_source):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif os.path.exists(image_source):
        image = Image.open(image_source).convert("RGB")

    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")

    # segment-anything model
    # sam = sam_model_registry["vit_b"](checkpoint="/Users/agupta/Documents/parallel-processing-using-k8s/model1/sam_vit_b_01ec64.pth").to("cuda")
    # predictor = SamPredictor(sam)
    # image = cv2.imread(image)
    # predictor.set_image(image)


    inputs = processor(images=image, text=[query], return_tensors="pt")

    with torch.no_grad():
            outputs = model(**inputs)

    logits = outputs.logits  # Classification scores
    # boxes = outputs.pred_boxes  # Bounding boxes
    boxes = outputs.pred_boxes[0].detach().numpy()

    # Convert logits to confidence scores
    scores = torch.sigmoid(outputs.logits)[0].detach().numpy()  # Convert logits to probabilities

    # print("Scores:", scores)
    # print("Bounding Boxes:", boxes)

    # Filter results based on a confidence threshold
    threshold = 0.1
    # detected_objects = [(box, score) for box, score in zip(boxes, scores) if score > threshold]
    detected_objects = [(box, score) for box, score in zip(boxes, scores) if score.max() > threshold]

    print({'prediction' : detected_objects[0][0], 'confidence' : detected_objects[0][1]})
    return {'prediction' : detected_objects[0][0], 
            'confidence' : detected_objects[0][1]}

    # bbox = []
    # Print results
    # if detected_objects:
    #     # print(f"Detected {len(detected_objects)} objects for query '{query}':")
    #     for i, (box, score) in enumerate(detected_objects):
    #     # print(f"  Object {i+1}: Box {box}, Confidence {score.max():.2f}")
    #     # segment-anything
    #     #bbox = box
    #     # segment-anything
    #     #masks, _, _ = predictor.predict(box=bbox, multimask_output=True)

    #     # Optionally visualize the result using OpenCV
    #     #cv2.imshow("Segmented Image", masks[0])  # Show the first mask
    # else:
    #     print("No objects detected with sufficient confidence.")



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_owl_v2.py <image_path_or_url>")
        sys.exit(1)

    image_source = sys.argv[1]
    query = sys.argv[2]
    text_labels = [[query.split(",")]]
    owlv2_results(image_source, query)