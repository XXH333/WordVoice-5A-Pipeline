# pip install num2words
import re
from num2words import num2words
from typing import Match
from collections import defaultdict
# from wetext import Normalizer
# normalizer = Normalizer(lang="en", operator="tn")  # 英文文本的

ORDINAL_MAP = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "5th": "fifth",
    "21st": "twenty first",
    "22nd": "twenty second",
    "31st": "thirty first"
}

# 最终修正后的正则表达式：更强大地处理整数、带逗号的整数和浮点数
# 匹配逻辑: (包含数字和逗号的串 + 可选的小数部分) OR (只包含小数部分)
NUMBER_REGEX = re.compile(r'\b[\d,]+(?:\.\d+)?\b|(?<!\w)\.\d+\b')

def _float_to_words(number_str: str) -> str:
    """
    内部函数：将数字字符串转换为英文单词，并确保小数部分逐位朗读。
    （保持与前一个版本一致，因为它逻辑上是正确的）
    """
    original_number = number_str 
    
    try:
        # 清理逗号
        cleaned_number_str = number_str.replace(',', '')
        
        # 1. 如果是纯整数
        if '.' not in cleaned_number_str:
            integer_val = int(cleaned_number_str)
            return str(num2words(integer_val, lang='en'))

        # 2. 处理浮点数
        if cleaned_number_str.startswith('.'):
            standardized_str = '0' + cleaned_number_str
        elif cleaned_number_str.endswith('.'):
             standardized_str = cleaned_number_str + '0'
        else:
            standardized_str = cleaned_number_str
        
        integer_part, dot, decimal_part = standardized_str.partition('.')

        if not integer_part:
            integer_val = 0
        else:
            integer_val = int(integer_part)
            
        integer_words = str(num2words(integer_val, lang='en'))
            
        point_word = "point"

        # 2b. 转换小数部分 (严格逐位朗读)
        decimal_words = []
        if not decimal_part:
             decimal_words.append("zero")
        else:
            for digit in decimal_part:
                if digit.isdigit():
                    decimal_words.append(str(num2words(int(digit), lang='en')))
        
        # 3. 拼接结果
        return f"{integer_words} {point_word} {' '.join(decimal_words)}"

    except Exception as e:
        print(f"[ERROR] Failed to normalize '{original_number}': {e}. Returning original.")
        return original_number 


def split_mixed_token(match):
    token = match.group(0)

    # 只拆 token 内部的数字
    token = re.sub(r'([A-Za-z])(\d)', r'\1 \2', token)
    token = re.sub(r'(\d)([A-Za-z])', r'\1 \2', token)

    # 如果 token 里还有纯数字（比如 A19301s），再拆数字
    token = re.sub(r'\d{5,}', lambda m: " ".join(m.group(0)), token)

    return token

def parse_control_text(input_text: str):
    """
    解析包含控制信息的文本。
    键仅支持: eng, pit, dur, bnd, ton
    值支持: 数字 (整数/浮点数) 或 单词 (字符串)
    """
    # 正则表达式更新：
    # (eng|pit|dur|bnd|ton) 限定了支持的 key
    # ([a-zA-Z0-9\.\-]+) 匹配数字（含负号和小数点）或英文字母组合
    control_pattern = re.compile(r"^\[\'(eng|pit|dur|bnd|ton)\'\:\s*([a-zA-Z0-9\.\-]+)\]$")
    
    cleaned_words = []
    # 【修改1】内层数据结构改为 dict，以完美匹配推理代码的格式
    control_dict = defaultdict(dict)
    
    tokens = input_text.strip().split()
    current_word_idx = -1
    
    for token in tokens:
        match = control_pattern.match(token)
        if match:
            key = match.group(1)
            value_str = match.group(2)
            
            # 智能类型转换：尝试 int -> float -> str
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str # 如果不是数字，则保留为单词字符串
            
            # 确定该控制信息作用的字索引
            target_idx = max(0, current_word_idx)
            # 【修改2】直接以字索引为键，控制参数为值
            control_dict[key][target_idx] = value
        else:
            # 如果是普通字，索引 +1，并加入纯文本列表
            current_word_idx += 1
            cleaned_words.append(token)
            
    cleaned_text = " ".join(cleaned_words)
    # 将 defaultdict 转回普通 dict 返回
    return cleaned_text, dict(control_dict)

def english_text_normalization(text: str) -> str:
    """
    主函数：对英文文本进行数字正则化处理。
    """
    # ==========================================
    # 新增 1：保护特殊插入符 (如 [*], [LAUGH] 等)
    # ==========================================
    tags = []
    def tag_replacer(match):
        tags.append(match.group(0))
        # 将索引数字转为纯字母 (0->a, 1->b, 12->bc)，防止被 mix_pattern 拆分
        # 例如第一个标签变成 zzmaskazz，第二个变成 zzmaskbzz
        alpha_idx = "".join(chr(97 + int(d)) for d in str(len(tags) - 1))
        return f" zzmask{alpha_idx}zz "

    # 匹配方括号及其中间的所有内容，并替换为纯字母占位符
    text = re.sub(r'\[.*?\]', tag_replacer, text)
    # ==========================================

    # 文本规范化
    text = text.lower() # 转为小写
    # 特殊符号处理
    text = re.sub(r'\s+,', ',', text) # 删除,前空格
    text = re.sub(r',+', ',', text) # 合并,
    text = re.sub(r',(?!\d)', ', ', text) # ,后续不是数字，加空格
    text = re.sub(r'\s+&\s+', ' and ', text) # & 替换为 and
    text = re.sub(r'(\d+)%', r'\1 percent ', text) # 数字后跟%替换为 percent

    # 处理数字
    # 特殊数字处理（1st、2nd、3rd）
    for k, v in ORDINAL_MAP.items():
        text = text.replace(f" {k} ", f" {v} ")
    # 处理 四位年份+s
    text = re.sub(r'\b(\d{2})(\d{2})s\b', r'\1 \2 ', text)
    # 处理字母＋长数字组合
    mix_pattern = re.compile(r'\b(?=[A-Za-z]*\d|\d*[A-Za-z])[A-Za-z0-9]+\b')
    text = mix_pattern.sub(split_mixed_token, text)

    # 统一处理数字
    def replace_match(match: Match) -> str:
        """供 re.sub 调用的替换函数。"""
        number_token = match.group(0)
        return _float_to_words(number_token)
        
    # 使用正则表达式的 sub 方法，查找所有匹配项并用 replace_match 的结果替换
    normalized_text = NUMBER_REGEX.sub(replace_match, text)

    # 后处理
    normalized_text = re.sub(r"[^\w\s']|_", ' ', normalized_text) # 标点替换为空格
    normalized_text = re.sub(r"(?<!\w)'|'(?!\w)", "", normalized_text) # 处理单引号

    # ==========================================
    # 新增 2：恢复特殊插入符
    # ==========================================
    for i, tag in enumerate(tags):
        alpha_idx = "".join(chr(97 + int(d)) for d in str(i))
        placeholder = f"zzmask{alpha_idx}zz"
        normalized_text = normalized_text.replace(placeholder, tag)
    # ==========================================

    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip() # 多余空格归一化
    normalized_text, control_dict = parse_control_text(normalized_text)

    return normalized_text, control_dict

# --- 示例使用 ---
if __name__ == "__main__":
    test_sentences = [
        "1st."
        "The price is y1 1081.123 dollars.",
        "We need 5000 units, not 1234s.",
        "The rate is just .015, which is low.",
        "The count is 5,000,000.",
        "The small number is 0.05 and the big one is 1,234,567.89.",
        "Today is [bnd:b0]2025s. 3am, ab1345s,[eng: -1] 21312th"
    ]

    print("--- 原始文本 vs 规范化文本 (最终修复版) ---")
    for sentence in test_sentences:
        normalized, control_dict = english_text_normalization(sentence) # mynorm
        # normalized = normalizer.normalize(sentence)  # wetext
        print(f"原始: {sentence}")
        print(f"规范: {normalized}")
        print(f"控制信息: {control_dict}\n")
        