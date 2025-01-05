# ModeLex

## Install

```bash
pip install git+https://github.com/andrew264/modelex.git
```

## Create Tokenized Dataset

To tokenize and save the dataset into a Parquet file, run:

```bash
modelex prepare_dataset --file {outputfile.parquet} --datasets classname:arg1,arg2
```

### Example Class: `modelex.data.Conversation`

This class accepts the following arguments:

- `data_path`: A directory containing a collection of JSON files with the format:
  
    ```json
    [
        {"user": "username", "message": "message from user"},
        {"user": "assistant", "message": "reply to user"},
        ...
    ]
    ```

- `tokenizer.json`: Path to the tokenizer configuration file.

## Train Models

To train a model
- First set up the config at `model_directory/trainer_config.yaml` see example [trainer_config.yaml](modelex/examples/trainer_config.yaml)
- and run the following command
```bash
modelex train model_directory/
```

## Run Models

- To run the model in chat mode, use:

    ```bash
    modelex prompt model_directory/
    ```

## Configuration Files

Configuration files are stored as `.yaml` files in `model_directory/`:

- `model.yaml`: Contains model hyperparameters.
- `trainer_config.yaml`: Contains training configurations such as optimizer, learning-rate, batch size.
- `sysprompt.txt`: Stores the system prompt.

Refer to `modelex/models/llm/config.py`, `modelex/training/trainer_config.py` for more details.

## History

This project was originally hosted at [andrew264/SmolLM](https://github.com/andrew264/Smol-LM), but the codebase became too messy, leading to this major rewrite.

## Future Plans

The goal is to build a multimodal LLM that supports any-to-any data modality. Experimentation for this is ongoing at:

- [ImageExpts](https://github.com/andrew264/ImageExpts)
- [AudioExpts](https://github.com/andrew264/AudioExpts)
