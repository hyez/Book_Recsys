# Book Recommender Systems

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-daisy) [![Version](https://img.shields.io/badge/version-v1.0.0-orange)](https://github.com/HYEZ/Book_Recsys) ![GitHub repo size](https://img.shields.io/github/repo-size/HYEZ/Book_Recsys)

## Platform
- python: 3.5+
- Pytorch: 1.0+

## Model
<img width="1931" alt="Model" src="https://user-images.githubusercontent.com/21326503/152167939-30202625-b354-4980-a887-7c5459b67bc7.png">


## How to run

### Training
```python
python maml.py
```
You can modify the detailed parameters according to the definition in maml.py.

### Testing
```python
python maml.py --test
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
