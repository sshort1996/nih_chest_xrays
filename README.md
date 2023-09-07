# nih_chest_xrays
Sample dataset downloaded from [kaggle dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample)

# Repository structure

```
NIH_CHEST_XRAYS
└── MNIST/
└───── CNN/
├──────── compile.py
├──────── preProcessor.py
├──────── test.py
└──────── train.py
└───── expl_nb.ipynb
└───── data
├──────── classic
│       ├── archive.zip
│       ├── t10k-images-idx3-ubyte
│       │   └── t10k-images-idx3-ubyte
│       ├── t10k-images.idx3-ubyte
│       ├── t10k-labels-idx1-ubyte
│       │   └── t10k-labels-idx1-ubyte
│       ├── t10k-labels.idx1-ubyte
│       ├── train-images-idx3-ubyte
│       │   └── train-images-idx3-ubyte
│       ├── train-images.idx3-ubyte
│       ├── train-labels-idx1-ubyte
│       │   └── train-labels-idx1-ubyte
│       └── train-labels.idx1-ubyte
└──────── fashion
        ├── archive.zip
        ├── fashion-mnist_test.csv
        ├── fashion-mnist_train.csv
        ├── t10k-images-idx3-ubyte
        ├── t10k-labels-idx1-ubyte
        ├── train-images-idx3-ubyte
        └── train-labels-idx1-ubyte

└── NIH/
└───── CNN/
├──────── compile.py
├──────── preProcessor.py
├──────── test.py
└──────── train.py
└───── expl_nb.ipynb
└───── data/
├──────── images
├──────── test_nih
│           └── images
                ...
            └── labels.csv
└──────── training_nih
            └── images
                ...
            └── labels.csv
└── testing/
│   ├── images
│   ├── test_nih
│       └── images
│   └── training_nih
│       └── images
└── pages/
└── Home.py
```