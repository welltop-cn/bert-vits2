import os
import re
import shutil


def extract_content_within_parentheses(text):
    # 正则表达式匹配括号内的内容
    matches = re.findall(r'\((.*?)\)', text)
    return matches

for x in os.listdir("./"):
    if x.startswith('12'):
        num = extract_content_within_parentheses(x)[0]
        for item in os.listdir(f"./{x}"):
            if item.endswith("WAV"):
                shutil.move(f"./{x}/{item}", f"./{num}.wav")
            else:
                shutil.move(f"./{x}/{item}", f"./{num}.srt")
