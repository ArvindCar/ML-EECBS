## To run EECBS
Navigate to ML-EECBS directory and run

```python run_experiments.py --instance instances/<instance>.txt --solver EECBS```

Example:

```python run_experiments.py --instance instances/test_6.txt --solver EECBS```

## To collect Data using EECBS

```python run_experiments.py --instance instances/<instance>.txt --solver EECBSDC --map_name <map_name>```

## To train the SVR model

```python SVR.py --map_name <map_name>```

## To run ML-EECBS with the trained model
```python run_experiments.py --instance instances/<instance>.txt --solver MLEECBS --map_name <map_name>```
