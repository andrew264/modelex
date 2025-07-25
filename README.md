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

### Example Class: `modelex.data.Conversations`

This class accepts the following arguments:

- `data_path`: A directory containing a collection of JSON files with the format:
  
    ```json
    [
        {"role": "user", "content": [{"type":  "text", "text":"message from user"}]},
        {"role": "assistant", "content": [{"type":  "reason", "text":"reasoning for the query"}, {"type":  "text", "text":"reply to user"}]},
        ...
    ]
    ```

- `path`: Path to the json files
- `tokenizer_path`: Path to the tokenizer.json file.
- `chat_format`: Chat format to use ["llama3", "chatml", "custom"]
- `has_reasoning`: boolean value. weather to use reasoning tokens, must be in the JSON files with `reasoning` key

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
    modelex chat model_directory/
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
