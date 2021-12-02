# Book Recommender Systems


## Platform
- python: 3.5+
- Pytorch: 1.0+

## Model
![Model](https://user-images.githubusercontent.com/21326503/144397664-8223390c-85d9-485a-8e4e-0fc879865323.png)

We offer the training process and model in multi_result_files/9b8290dd3f63cbafcd141ba21282c783.pkl.

## How to run

### Training
```python
python3 maml.py
```
You can modify the detailed parameters according to the definition in maml.py.

### Testing
```python
python3 maml.py --test
```
By default, you can directly add the test argument to test the model obtained from the same aruguments setting.
```python
mode_path = utils.get_path_from_args(args)
```
You can also modify the code in maml.py manually since the arguments may vary for training and testing process.
```python
mode_path = '9b8290dd3f63cbafcd141ba21282c783'
```



# Acknowledgement.
This code refers code from:
[waterhorse1/MELU_pytorch](https://github.com/waterhorse1/MELU_pytorch).
