# ECE 9000
 
This Retrieval System is built using Flask framework and utilizes a pre-trained model trained on the Market1501 dataset. The implementation and training process of the model can be found at https://github.com/21blurryface/ECE9000-Model.

To use the system, first install PyTorch and Flask, put the images into data folder and model into model folder, then run python start.py. Note that using a GPU can reduce retrieval time, but it is not required. 

The length of the hash code in this task is set to 256 bits to represent the differences among nearly 1000 classes of pedestrians in the dataset.

The system and model are trained on the Market1501 dataset for person re-identification but can be adapted to other similar tasks.

Thank you for using our Retrieval System. We hope it contributes to the advancement of artificial intelligence and related fields.
