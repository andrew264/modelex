from typing import TypedDict

class Message(TypedDict):
    user: str
    message: str

class Prompt:
    BOT, EOT = '<|begin_of_text|>', '<|eot_id|>\n'
    SH, EH = '<|start_header_id|>', '<|end_header_id|>\n\n'
    def __init__(self, assistant_name: str, sysprompt: str) -> None:
        self.assistant_name = assistant_name
        self.sysprompt = sysprompt
        self.msgs: list[Message] = []
    def add_msg(self, user: str, msg: str)->None: self.msgs.append(Message(user=user, message=msg))
    def add_msgs(self, msgs: list[Message])->None: self.msgs.extend(msgs)
    def reset(self)->None: self.msgs = []
    def get_prompt_for_completion(self) -> str:
        out = f'{self.BOT}{self.SH}system{self.EH}{self.sysprompt}{self.EOT}'
        for m in self.msgs: out += f'{self.SH}{m["user"]}{self.EH}{m["message"]}{self.EOT}'
        out += f'{self.SH}{self.assistant_name}{self.EH}'
        return out