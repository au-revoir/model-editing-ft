# Command

## Single Edit
### Counterfact
To train and calculate the ES, PS, NS, Fluency and Consistency run:
```
bash execute.sh
```
There will be individual result .json files stored in separate directories for each GPU inside ```results```. Taking the mean of each of the metrics will give the final result.

You can edit the number of GPUs in ```execute.sh```. Check ```logs``` directory for any errors.
