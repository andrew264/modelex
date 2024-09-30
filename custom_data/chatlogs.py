import datetime
import glob
import json
import os
from typing import Dict, List, TypedDict

from tokenizers import Tokenizer

class Message(TypedDict):
    user: str
    message: str

class Conversations:
    CROSS_ENTROPY_IGNORE_IDX = -100
    BOT, EOT = '<|begin_of_text|>', '<|eot_id|>\n'
    SH, EH = '<|start_header_id|>', '<|end_header_id|>\n\n'
    def __init__(self, path: str, tokenizer_path: str) -> None:
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._files = glob.glob(f'{path}/**/*.json', recursive=True)
        self._sysprompt = ''
        dt = datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
        with open(os.path.join(path, 'sysprompt.txt'), 'r', encoding='utf-8') as f: self._sysprompt = f.read().format(datetime=dt).strip()
        self._assistant_name = 'assistant'
        with open(os.path.join(path, 'name'), 'r', encoding='utf-8') as f: self._assistant_name = f.read()
    def __len__(self,)->int: return len(self._files)
    def _get_encoded(self, data: List[Message]) -> Dict[str, List[int]]:
        ids, labels = [], []
        sp = self._tokenizer.encode(f'{self.BOT}{self.SH}system{self.EH}{self._sysprompt}{self.EOT}', add_special_tokens=False)
        ids.extend(sp.ids)
        labels.extend([self.CROSS_ENTROPY_IGNORE_IDX] * len(sp.ids))
        for msg in data:
            u = self._tokenizer.encode(f'{self.SH}{msg["user"]}{self.EH}', add_special_tokens=False)
            m = self._tokenizer.encode(f'{msg["message"]}{self.EOT}', add_special_tokens=False)
            combined = u.ids+m.ids
            ids.extend(combined)
            if msg['user'] == self._assistant_name: labels.extend(([self.CROSS_ENTROPY_IGNORE_IDX] * len(u.ids)) + m.ids)
            else: labels.extend([self.CROSS_ENTROPY_IGNORE_IDX] * len(combined))
        return {'input_ids': ids, 'labels': labels}
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        try:
            with open(self._files[idx], 'r', encoding='utf-8') as f: data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error in file {self._files[idx]}: {e}")
            raise e
        return self._get_encoded(data)
