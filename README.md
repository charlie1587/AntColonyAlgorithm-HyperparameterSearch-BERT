# Ant Colony Algorithm-Based Hyperparameter Search for BERT Classifiers


## Environment 

### Hardware 
CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz

GPU: NVIDIA RTX A6000 48GB CUDA 12.2

### Software
python                    3.9.18 

torch                     2.1.1 

torchvision               0.16.1 


## Dataset

https://github.com/allenai/scibert/tree/master/data/text_classification/chemprot

## Usage

Run the following command to train the model:

```bash
python main.py
```
The num_ants and iterations dramatically affect the performance of the ant colony algorithm. You should set the num_ants and the iterations carefully in `config.py`.
The two hyperparameters of BERT classifier are the lr and the scheduler_gamma. You should set  `lower_bound` and `upper_bound` in `config.py`.  

File structure:

ant.py: the main file of the ant colony algorithm

classifier.py: the classifier of BERT

dataset.py: the dataset of the classifier

config.py: the configuration of the ant colony algorithm

main.py: the main file
