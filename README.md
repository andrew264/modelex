# ModeLex

## Create Tokenized Dataset

To tokenize and save the dataset into a Parquet file, run:

```bash
scripts/prepare_datasets.py --file {outputfile.parquet} --datasets classname:arg1,arg2
```

### Example Class: `custom_data.chatlogs.Conversation`

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
- First set up the config at `model_directory/train.yaml`
- and run the following command
```bash
python scripts/train.py model_directory/ train.parquet validate.parquet
```

## Run Models

- To run the model in chat mode, use:

    ```bash
    scripts/prompt.py model_directory/
    ```

- To run the model in sentence completion mode, use:

    ```bash
    scripts/generate.py model_directory/
    ```

## Configuration Files

Configuration files are stored as `.yaml` files in `model_directory/`:

- `model.yaml`: Contains model hyperparameters.
- `train.yaml`: Contains training hyperparameters.
- `peft.yaml`: Contains performance-efficient fine-tuning (PEFT) configurations such as LoRA.
- `inference.yaml`: Contains model generation parameters such as `chat_format`, `top_p`, `eos_tokens`, etc.
- `sysprompt.txt`: Stores the system prompt.

Refer to `models/config.py` for more details.

## History

This project was originally hosted at [andrew264/SmolLM](https://github.com/andrew264/Smol-LM), but the codebase became too messy, leading to this major rewrite.

## Future Plans

The goal is to build a multi-model LLM that supports any-to-any data modality. Experimentation for this is ongoing at:

- [ImageExpts](https://github.com/andrew264/ImageExpts)
- [AudioExpts](https://github.com/andrew264/AudioExpts)
