# TDGAT
This is the implementation of TDGAT (Temporal-aware Dynamic Graph Neural Network for Product Search).

### Build environment
The requirement of our model is illustrated in ```requirements.txt```. You can run following code to download of the dependency packages.

```
pip install -r requirements.txt
```

### Download data
Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/ (In our paper, we used 5-core data) and change the paths of these files in ```utils.py```.

### Generate data
Run the file ```gen_data_build_graph.py``` to generate the data and build graph prepared for our model. The detailed explanations of parameters are shown in the file.

```
python gen_data_build_graph.py --dataset CELLPHONE_DATA --item_max_length 20 --user_max_length 20 --job 10 --k_hop 2
```

### Training and Testing 
Then you can run the file ```main.py``` to train and test our model. The detailed explanations of parameters are shown in the file.

```
python -u main.py --dataset=CELLPHONE_DATA --hidden_size=50 --epoch=20 --item_max_length=20 --user_max_length=20
```