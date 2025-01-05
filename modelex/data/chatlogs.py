import datetime
import glob
import json
import os
from typing import Dict, List, Optional, TypedDict, Union

import numpy as np
from tokenizers import Tokenizer

from modelex.data.prompt_format import ChatFormat, ChatFormatFactory, ChatFormatType
from modelex.utils import str2bool

class Message(TypedDict):
    user: str
    message: str
    thoughts: Optional[str]

class Conversations(ChatFormat):
    CROSS_ENTROPY_IGNORE_IDX = -100
    def __init__(self, path: str, tokenizer_path: str, chat_format: str = "llama3", enable_thoughts: Union[bool, str] = False) -> None:
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._files = glob.glob(f'{path}/**/*.json', recursive=True)
        self._sysprompt = ''
        dt = datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
        with open(os.path.join(path, 'sysprompt.txt'), 'r', encoding='utf-8') as f: self._sysprompt = f.read().format(datetime=dt).strip()
        self._assistant_name = 'assistant'
        with open(os.path.join(path, 'name'), 'r', encoding='utf-8') as f: self._assistant_name = f.read()
        self._apply_format(chat_format)
        if isinstance(enable_thoughts, str):
            self.enable_thoughts = str2bool(enable_thoughts)
        else:
            self.enable_thoughts = enable_thoughts
    def _apply_format(self, chat_format: str):
        fmt = ChatFormatFactory.create(ChatFormatType(chat_format))
        self.BOT, self.EOT = fmt.BOT, fmt.EOT
        self.SH, self.EH = fmt.SH, fmt.EH
    def __len__(self, ) -> int: return len(self._files)
    def _get_encoded(self, data: List[Message]) -> Dict[str, np.ndarray]:
        ids, labels = [], []
        sp = self._tokenizer.encode(f'{self.BOT}{self.SH}system{self.EH}{self._sysprompt}{self.EOT}'.strip(), add_special_tokens=False)
        ids.extend(sp.ids)
        labels.extend([self.CROSS_ENTROPY_IGNORE_IDX] * len(sp.ids))
        for msg in data:
            u = self._tokenizer.encode(f'{self.SH}{msg["user"]}{self.EH}', add_special_tokens=False)
            if self.enable_thoughts:
                t: str = msg.get('thoughts', '')
                t = f'<think>{t.strip()}</think>\n'
            else:
                t = ''
            m = self._tokenizer.encode(f'{t}{msg["message"]}{self.EOT}', add_special_tokens=False)
            combined = u.ids + m.ids
            ids.extend(combined)
            if msg['user'] == self._assistant_name: labels.extend(([self.CROSS_ENTROPY_IGNORE_IDX] * len(u.ids)) + m.ids)
            else: labels.extend([self.CROSS_ENTROPY_IGNORE_IDX] * len(combined))
        return {'input_ids': np.array(ids, dtype=np.int32), 'labels': np.array(labels, dtype=np.int32)}
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        try:
            with open(self._files[idx], 'r', encoding='utf-8') as f: data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error in file {self._files[idx]}: {e}")
            raise e
        return self._get_encoded(data)