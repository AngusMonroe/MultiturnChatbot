# MultiturnChatbot

A multiturn seq2seq chatbot implement by pytorch with [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and [DuConv Dataset](https://ai.baidu.com/broad/introduction?dataset=duconv).

## Getting Started

This codebase is tested using Ubuntu 16.04, Python 3.6 and a single NVIDIA TITAN X GPU. Similar configurations are preferred.

### Installation

- Clone this repo:

    ```
    git clone https://github.com/AngusMonroe/MultiturnChatbot
    cd MultiturnChatbot
    ```

- Install requirements, run:

    ```
    pip install -r requirements.txt
    ```

### Prepare dataset

You could download [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and [DuConv Dataset](https://ai.baidu.com/broad/introduction?dataset=duconv) and put it under `./data/[dataset_name]`.

Put glove pre-trained word vectors file to `./model/glove`

### Preprocess

- Cornell Movie-Dialogs Corpus

    ```
    python preprocess_cornell.py
    ```

- Duconv

    ```
    python preprocess_duconv.py
    ```

### Train

Run with default config:

```
python train.py
```

```
$ python train.py
Start preparing training data ...
Reading lines...
Read 159942 sentence pairs
Trimmed to 33851 sentence pairs
Counting words...
Counted words: 48831
Building encoder and decoder ...
torch.Size([48831, 300])
Models built and ready to go!
Building optimizers ...
Starting Training!
Initializing ...
Training...
Iteration: 10; Percent complete: 0.2%; Average loss: 10.6792
...
Iteration: 4000; Percent complete: 100.0%; Average loss: 3.3271
Start testing...
100%|████████████████████████████████████████████████████████████████████████████████| 16108/16108 [02:58<00:00, 90.19it/s]
Done! The model name is: 20190805-131336
```

Get available parameters:

```
python train.py --help
```

```
$ python train.py --help
Usage: train.py [options]

Options:
  -h, --help            show this help message and exit
  -C CORPUS_NAME, --corpus_name=CORPUS_NAME
                        Corpus name (cornell-movie-dialogs-corpus or duconv)
  -F DATA_FILE, --data_file=DATA_FILE
                        Data file name
  -T TRAIN_FILE, --train_file=TRAIN_FILE
                        Train file name
  -E TEST_FILE, --test_file=TEST_FILE
                        Test file name
  -G GLOVE_PATH, --glove_path=GLOVE_PATH
                        Path of Glove word vector file
  -A ATTN_MODEL, --attn_model=ATTN_MODEL
                        Attention model
  -H HIDDEN_SIZE, --hidden_size=HIDDEN_SIZE
                        Dimensionality of the glove word vector
  --encoder_n_layers=ENCODER_N_LAYERS
                        Layer number of encoder
  --decoder_n_layers=DECODER_N_LAYERS
                        Layer number of decoder
  -D DROPOUT, --dropout=DROPOUT
                        Dropout
  -B BATCH_SIZE, --batch_size=BATCH_SIZE
                        Batch size
  --clip=CLIP           Clip gradients: gradients are modified in place
  --teacher_forcing_ratio=TEACHER_FORCING_RATIO
                        Teacher forcing ratio
  --decoder_learning_ratio=DECODER_LEARNING_RATIO
                        Decoder learning ratio
  -L LEARNING_RATE, --learning_rate=LEARNING_RATE
                        Learning rate
  -N N_ITERATION, --n_iteration=N_ITERATION
                        Epoch number
  -P PRINT_EVERY, --print_every=PRINT_EVERY
                        Print frequency
  -S SAVE_EVERY, --save_every=SAVE_EVERY
                        Save frequency
```

### Evaluate

The evaluation metrics are given in the form of BLEU1 and BLEU2.

```
python test.py [test_file]
```

```
16108
16108
0.12676846929645086
0.005143676257318174
```

### Start a service for case study

```
python service.py [model_path]
```

```
Models built and ready to go!
Start preparing training data ...
Reading lines...
Read 159942 sentence pairs
Trimmed to 33851 sentence pairs
Counting words...
Counted words: 48831
>
```

```
> 你 喜欢 看 电影 么 ？
Bot: 喜欢 啊 ， 有 推荐 吗 ？
>
```

## Reference

https://github.com/baidu/knowledge-driven-dialogue/blob/master/task_description.pdf

Papineni K, Roukos S, Ward T, et al. BLEU: a method for automatic evaluation of machine translation[C]//Proceedings of the 40th annual meeting on association for computational linguistics. Association for Computational Linguistics, 2002: 311-318.
