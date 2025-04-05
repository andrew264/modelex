import datetime
import glob
import json
import os
from typing import Dict, List, Union

import numpy as np
from tokenizers import Tokenizer

from modelex.utils import str2bool
from modelex.utils.conversation_format import ChatFormatFactory, ChatFormatType, Conversation, ChatFormat

class Conversations:
    CROSS_ENTROPY_IGNORE_IDX = -100
    def __init__(self, path: str, tokenizer_path: str, chat_format: str = "llama3", has_reasoning: Union[bool, str] = False) -> None:
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._files = glob.glob(f'{path}/**/*.json', recursive=True)
        self._sysprompt_default = ''
        dt = datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
        with open(os.path.join(path, 'sysprompt.txt'), 'r', encoding='utf-8') as f: self._sysprompt_default = f.read().format(datetime=dt).strip()
        self._assistant_name = 'assistant'
        with open(os.path.join(path, 'name'), 'r', encoding='utf-8') as f: self._assistant_name = f.read()
        self._apply_format(chat_format)
        if isinstance(has_reasoning, str):
            self.has_reasoning = str2bool(has_reasoning)
        else:
            self.has_reasoning = has_reasoning
    def _apply_format(self, chat_format: str):
        fmt = ChatFormatFactory.create(ChatFormatType(chat_format))
        self.BOT, self.EOT = fmt.BOT, fmt.EOT
        self.SH, self.EH = fmt.SH, fmt.EH
    def __len__(self, ) -> int: return len(self._files)
    def _get_encoded(self, data: Conversation) -> Dict[str, np.ndarray]:
        inputs, labels = [], []
        if data[0].get('role') == 'system':
            sysprompt = ''
            for row in data[0]['content']:
                sysprompt += row['text'] + '\n'
            data.pop(0)
        else:
            sysprompt = self._sysprompt_default
        sp = self._tokenizer.encode(f'{self.BOT}{self.SH}system{self.EH}{sysprompt}{self.EOT}'.strip(), add_special_tokens=False)
        inputs.extend(sp.ids)
        labels.extend([self.CROSS_ENTROPY_IGNORE_IDX] * len(sp.ids))
        for msg in data:
            u = self._tokenizer.encode(f'{self.SH}{msg["role"]}{self.EH}', add_special_tokens=False)
            t = ''
            for row in msg['content']:
                if row['type'] == 'reason':
                    if self.has_reasoning:
                        t += f'<think>{row["text"]}</think>\n'
                    else: continue
                else:
                    t += row['text'] + '\n'
            m = self._tokenizer.encode(f'{t}{self.EOT}', add_special_tokens=False)
            combined: List[int] = u.ids + m.ids
            inputs.extend(combined)
            if msg['role'] == self._assistant_name: labels.extend(([self.CROSS_ENTROPY_IGNORE_IDX] * len(u.ids)) + m.ids)
            else: labels.extend([self.CROSS_ENTROPY_IGNORE_IDX] * len(combined))
        return {'input_ids': np.array(inputs, dtype=np.int32), 'labels': np.array(labels, dtype=np.int32)}
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        try:
            with open(self._files[idx], 'r', encoding='utf-8') as f: data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error in file {self._files[idx]}: {e}")
            raise e
        return self._get_encoded(data)