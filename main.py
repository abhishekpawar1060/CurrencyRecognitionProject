import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

model = load_model('currency_classification_model.h5')

currency_labels = {
    '1Hundrednote': '100rs',
    '2Hundrednote': '200rs',
    '2Thousandnote': '2000rs',
    '5Hundrednote': '500rs',
    'Fiftynote': '50rs',
    'Tennote': '10rs',
}

testing_folder = 'dataset/Test/'

def predict_and_display(image_path):
    processed_image = preprocess_image(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)  
    
    prediction = model.predict(processed_image)
    predicted_currency_index = np.argmax(prediction)  
    confidence = prediction[0][predicted_currency_index] * 100  
    
    # Get the predicted currency label
    currency_folder = os.path.basename(os.path.dirname(image_path))
    predicted_currency_label = currency_labels[currency_folder]
    
    image = cv2.imread(image_path)
    
    cv2.putText(image, predicted_currency_label, (50, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.imshow('Predicted Currency', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(f"Predicted currency: {predicted_currency_label}, Predicated Accuracy: {confidence:.2f}%")
    print("Predicated Currency",predicted_currency_label)

image_path = "dataset/Test/Fiftynote/27.jpg"   # Input path of the image
predict_and_display(image_path)
















# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Function to load and preprocess image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (224, 224))  # Resize image to match model input size
#     img = img / 255.0  # Normalize pixel values to [0, 1]
#     return img

# # Load the trained model
# model = load_model('currency_classification_model.h5')

# # Dictionary to map currency folder names to labels
# currency_labels = {
#     '1Hundrednote': '100rs',
#     '2Hundrednote': '200rs',
#     '2Thousandnote': '2000rs',
#     '5Hundredsnote': '500rs',
#     'Fiftynote': '50rs',
#     'Tennote': '10rs',
# }

# # Path to the testing folder
# testing_folder = 'dataset/Test/'

# # Iterate through folders in the testing folder
# for currency_folder in os.listdir(testing_folder):
#     currency_folder_path = os.path.join(testing_folder, currency_folder)
    
#     # Iterate through images in the currency folder
#     for image_name in os.listdir(currency_folder_path):
#         image_path = os.path.join(currency_folder_path, image_name)
        
#         # Preprocess the image
#         processed_image = preprocess_image(image_path)
#         processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        
#         # Make prediction
#         prediction = model.predict(processed_image)
#         predicted_currency = np.argmax(prediction)  # Get the index of the highest probability
        
#         # Get the predicted currency label
#         predicted_currency_label = currency_labels[currency_folder]
        
#         # Print the prediction along with the image name
#         print(f"Predicted currency for {image_name}: {predicted_currency_label}")
        
#         # Display the image
#         cv2.imshow(image_name, cv2.imread(image_path))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

