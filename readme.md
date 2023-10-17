# Code implementation for "A dual attention LSTM lightweight model based on exponential smoothing for remaining useful life prediction"
This project provides all the code for data-preprocessing, model construction and model evaluation. Checkpoints for the trained models are also included for further study. Pytorch is used to construct the DA-LSTM model.

## File Structure
```bash
├───dataset
│   ├───__init__.py
│   └───preprocessing.py
├───model
│   ├───__init__.py
│   ├───Attention_modules.py
│   └───LSTM_Attention.py
├───scipt
│   ├───parametric_statistics.py
│   ├───test_model.py
│   └───train_model.py
├───trials
│   ├───model_FD001.pkl
│   ├───model_FD002.pkl
│   ├───model_FD003.pkl
│   └───model_FD004.pkl
├───utils
│   ├───__init__.py
│   └───functions.py
```
- The `CMAPSSData/` folder contains the official CMAPSS-dataset.
- The `dataset/` folder and `utils/` folder contain the functions for data preprocessing and model training respectively.
- The `trials/` folder contains the checkpoints for the trained models on the four sub-datasets.
- The `model/` folder contains the components for the DA-LSTM model.

## Running Guide
You can either load the models for testing or training your own models from scratch.
Take notice that there are four sub-datasets in the CMAPSS Dastset. Consequently, certain parameters need to be changed according to the subdatasets.
### Running Tests
You can load the trained checkpoints directly by running `test_model.py` in the `scipt/` folder. 
```bash
cd [parent folder of the project]/CMAPSS-release/scipt/
python .\test_model.py
```

### Training Models
If you need to retrain the models, you can run `train_model.py` in the `scipt/` folder. The trianed checkpoints are save in the `\trials` directory.
```bash
cd [parent folder of the project]/CMAPSS-release/scipt/
python .\train_model.py
```

### Evaluating Models
If you need to evaluate number of parameters in the model, you can run `parameter statistics.py` in the `scipt/` folder.

```bash
cd [parent folder of the project]/CMAPSS-release/scipt/
python .\parameter statistics.py
```

