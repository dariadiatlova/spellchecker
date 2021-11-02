# Spellchecker for English language

### Usage
Clone repository and run [run_cpellchecker](run_cpellchecker.py) script from the command line:


      python3 -m run_cpellchecker -w <misspelled_word> -i <path_to_the_config>
      

**Argument** | Value |
---|---|
-w, --word | An English word to check the spelling.
-i, --input_config_path | Optional argument, by default use [config.json file](config.json) that have 2 main arguments: path to the trained Catboost model and number of suggested words to return (default is 5). 


### Training 

An example of training and validation of Spellchecker can be found in the [jupyter notebook](notebook.ipynb), trained moel weights were dumped to the [repositiory folder](trained_model). 
 

### Inference mode

Spellchecker provides three possible outcomes:

- if the word is in [`hunnspell dictionary`](https://spylls.readthedocs.io/en/latest/hunspell/dictionary.html) – the message `Good job! The spelling is correct!` is returned;
- if `hunnspell` couldn't suggest any words that might be correct – `<input_word>` is returned;
- if `hunnspell` has multiple suggestions – trained model is used to extract features and rank candidates, the message `You've probably meant: <n-candidates>` is returned. Where `n-candidates` is a list of n (that can be configured in [config.json file](config.json)) words.

### Accuracy

Spellchecker was tested on [test set](http://aspell.net/test/cur/), evaluation results of the model, loaded to the repository are presented in the table below:

**Accuracy** | Score |
---|---|
Vanilla hunspell | 0.55
**Set of distances + Catboost ranking**
acc@1 | 0.37
acc@5 | 0.76
acc@10 | 0.84
