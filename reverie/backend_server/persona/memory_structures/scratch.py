"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: scratch.py
Description: Defines the short-term memory module for generative agents.
"""
import datetime
import json
import sys
sys.path.append('../../')

from global_methods import *
from deepseek_adapter import chat_completion   # <--- 加這行

class Scratch: 
    def __init__(self, f_saved): 
        # ...【原有初始化內容完全不變，略去】

        if check_if_file_exists(f_saved): 
            # ...【原有初始化內容完全不變，略去】
            pass

    # ...【所有原有 function 完全照舊，略去】

    # ================= 新增：Deepseek LLM 調用方法 =================
    def ask_llm(self, prompt, system_message=None, model="deepseek-chat", temperature=0.7, max_tokens=512):
        """
        用 Deepseek chat_completion 生成回應
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    # ===========================================================

    #（其他原有 function 完全不變）

    # 例如你可以咁用：
    # reply = self.ask_llm("你係邊個？用廣東話答我。")





















