# Facial-recognition-with-a-Siamese-network

One Shot Face-Recognition using Siamese Network

A Face Recognition Siamese Network implemented using Keras. Siamese Network is used for one shot learning which do not require extensive training samples for image recognition.


## Introduction

The process of learning good features for machine learning applications can be very computationally expensive and may prove difficult in
cases where little data is available. A prototypical example of this is the one-shot learning setting, in which we must correctly make predictions given only a single example of each new
class.
In this project , we  explore a method for
learning siamese neural networks which employ
a unique structure to naturally rank similarity between inputs. Once a network has been tuned,
we can then capitalize on powerful discriminative features to generalize the predictive power of
the network not just to new data, but to entirely
new classes from unknown distributions. 
## Datasets

LFW(Labeled Faces in the Wild) Datasets has been used for training the model
https://vis-www.cs.umass.edu/lfw/
## ARCHITECTURE
![alt text](image.png)!![alt text](image-3.png)
The Siamese network design comprises two identical subnetworks, each processing one of the inputs. Initially, the inputs undergo processing through a convolutional neural network (CNN), which extracts significant features from the provided images. These subnetworks then generate encoded outputs, often through a fully connected layer, resulting in a condensed representation of the input data.

The CNN consists of two branches and a shared feature extraction component, composed of layers for convolution, batch normalization, and ReLU activation, followed by max pooling and dropout layers. The final segment involves the FC layer, which maps the extracted features to the ultimate classification outcomes. A function delineates a linear layer followed by a sequence of ReLU activations and a series of consecutive operations (convolution, batch normalization, ReLU activation, max pooling, and dropout). The forward function guides the inputs through both branches of the network.

The Differencing layer serves to identify similarities between inputs and amplify distinctions among dissimilar pairs, accomplished using the Euclidean Distance function:
Distance(x₁, x₂) = ∥f(x₁) – f(x₂)∥₂

Two types of loss is used
1.The mathematical equation for Mean Absolute Error (MAE) or L1 Loss is:
MAE = (1/n) * Σ|yᵢ - ȳ|
2.The mathematical equation for Binary Cross-Entropy Loss, also known as Log Loss, is:
L(y, f(x)) = -[y * log(f(x)) + (1 - y) * log(1 - f(x))]

This property enables the network to acquire effective data representations apply that to fresh, unseen samples. Consequently, the network generates an encoding, often represented as a similarity score, that aids in-class differentiation.

Depict the network’s architecture in the accompanying figure. Notably, this network operates as a one-shot classifier, negating the need for many examples per class.

## Conclusion

In conclusion, the implementation of a One Shot Face-Recognition system utilizing Siamese Network architecture offers a promising solution to the challenges of face recognition tasks, particularly in scenarios with limited training data. By leveraging the Siamese Network's ability to learn subtle facial features and encode facial similarities, this approach demonstrates remarkable accuracy even with a single reference image per individual. Furthermore, its versatility extends to various applications, including security systems, access control, and personalized user experiences. As advancements continue in deep learning and computer vision, the integration of Siamese Networks into face recognition technology holds great potential for enhancing efficiency and reliability in real-world applications.
## Acknowledgements

 - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
 - https://www.youtube.com/playlist?list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH



