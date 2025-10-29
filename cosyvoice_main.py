#!/usr/bin/env python3
"""
CosyVoice 语音克隆示例
支持通过命令行参数传入待克隆的音频文件和输入文本
输出：音频文件 + 每个句子对应的起止时间（精确到毫秒）
"""

import torch
import torchaudio
from tqdm import tqdm
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice2
import argparse
import json
import sys
from pathlib import Path

# 添加第三方库路径
sys.path.append('third_party/Matcha-TTS')


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='CosyVoice 语音克隆示例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python cosyvoice_main.py --audio ./asset/zero_shot_prompt.wav 
    --text "你好，这是一个语音克隆测试"
  python cosyvoice_main.py --audio ./samples/voice_sample.wav 
    --text "今天天气真不错" --output ./output/cloned_speech.wav
  python cosyvoice_main.py --audio ./samples/voice_sample.wav 
    --text "今天天气真不错" --format srt --output ./output/cloned_speech.wav
        """
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='待克隆的音频文件路径 (支持wav格式)'
    )

    parser.add_argument(
        '--text',
        type=str,
        required=True,
        help='要合成的文本内容'
    )

    parser.add_argument(
        '--prompt_text',
        type=str,
        default='',
        help='提示文本，应该与音频文件中的内容匹配 (可选，留空则省略)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='pretrained_models/CosyVoice2-0.5B',
        help='模型路径 (默认: pretrained_models/CosyVoice2-0.5B)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./output/cloned_speech.wav',
        help='输出音频文件路径 (默认: ./output/cloned_speech.wav)'
    )

    parser.add_argument(
        '--timestamps_output',
        type=str,
        default='',
        help=('时间戳输出文件路径 '
              '(默认: 与音频文件同目录同名.json)')
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='生成次数 (默认: 1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='是否使用流式生成'
    )

    parser.add_argument(
        '--load_jit',
        action='store_true',
        help='是否加载JIT模型'
    )

    parser.add_argument(
        '--load_trt',
        action='store_true',
        help='是否加载TensorRT模型'
    )

    parser.add_argument(
        '--fp16',
        action='store_true',
        help='是否使用FP16精度'
    )

    parser.add_argument(
        '--max_sentence_length',
        type=int,
        default=50,
        help='最大句子长度，用于文本分割 (默认: 50)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'srt'],
        default='json',
        help='输出格式 (默认: json, 可选: srt)'
    )

    # 测试句子分割
    parser.add_argument(
        '--test_sentence_split',
        action='store_true',
        help='测试句子分割'
    )

    return parser.parse_args()


def validate_inputs(args):
    """验证输入参数"""
    # 检查音频文件是否存在
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {args.audio}")

    if not audio_path.suffix.lower() == '.wav':
        print(f"警告: 建议使用WAV格式音频文件，当前文件: {audio_path.suffix}")

    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {args.model_path}")

    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return True


def calculate_timestamps(audio_segments, sample_rate):
    """
    计算每个音频片段的时间戳（毫秒）

    Args:
        audio_segments: 音频片段列表
        sample_rate: 采样率

    Returns:
        timestamps: [(start_ms, end_ms), ...]
    """
    timestamps = []
    current_time_ms = 0.0

    for segment in audio_segments:
        # 计算当前片段的时长（毫秒）
        duration_ms = (segment.shape[1] / sample_rate) * 1000
        start_time_ms = current_time_ms
        end_time_ms = current_time_ms + duration_ms

        timestamps.append((start_time_ms, end_time_ms))
        current_time_ms = end_time_ms

    return timestamps


def is_only_punctuation(text):
    """
    检查文本是否只包含标点符号

    Args:
        text: 要检查的文本

    Returns:
        bool: 如果只包含标点符号返回True，否则返回False
    """
    if not text:
        return True

    # 定义所有标点符号
    punctuation_chars = {
        # 中文标点
        '。', '，', '、', '；', '：', '！', '？', '"', '"', ''', ''',
        '（', '）', '【', '】', '《', '》', '…', '——', '—', '……',
        # 英文标点
        '.', ',', ';', ':', '!', '?', '"', "'", '(', ')', '[', ']',
        '<', '>', '...', '--', '-', '/', '\\', '|', '@', '#', '$',
        '%', '^', '&', '*', '+', '=', '_', '~', '`'
    }

    # 检查每个字符是否都是标点符号
    for char in text:
        if char not in punctuation_chars and not char.isspace():
            return False

    return True


def custom_text_split(text, max_length=50):
    """
    自定义文本分割函数，优先使用句子级标点符号分割，过长时再使用智能分割

    Args:
        text: 输入文本
        max_length: 最大句子长度

    Returns:
        sentences: 分割后的句子列表
    """
    # 检查原文本是否只包含标点符号
    if is_only_punctuation(text):
        return []

    # 第一步：按句子级标点符号分割
    sentence_punctuation = ['。', '；', '？', '！', '.', ';', '?', '!']

    # 使用简单的字符串分割方法
    sentences = []
    current_sentence = ""

    i = 0
    while i < len(text):
        char = text[i]
        current_sentence += char

        # 如果遇到句子级标点符号
        if char in sentence_punctuation:
            # 保存当前句子
            sentence_to_save = current_sentence.strip()
            if sentence_to_save and not is_only_punctuation(sentence_to_save):
                sentences.append(sentence_to_save)
            current_sentence = ""

        i += 1

    # 处理最后剩余的文本
    if current_sentence.strip():
        final_sentence = current_sentence.strip()
        if not is_only_punctuation(final_sentence):
            sentences.append(final_sentence)

    # 如果没有分割出任何句子，按长度分割
    if not sentences:
        sentences = []
        for i in range(0, len(text), max_length):
            segment = text[i:i + max_length]
            if not is_only_punctuation(segment):
                sentences.append(segment)
        return sentences

    # 第二步：检查是否有句子过长，如果有则使用智能分割
    final_sentences = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            final_sentences.append(sentence)
        else:
            # 句子过长，使用智能分割
            smart_sentences = smart_split_long_sentence(sentence, max_length)
            final_sentences.extend(smart_sentences)

    return final_sentences


def smart_split_long_sentence(text, max_length):
    """
    智能分割过长的句子，避免单词被切碎

    Args:
        text: 要分割的文本
        max_length: 最大句子长度

    Returns:
        sentences: 分割后的句子列表
    """
    if len(text) <= max_length:
        return [text]

    sentences = []
    current_sentence = ""

    # 按空格分割单词
    words = text.split()

    for word in words:
        # 检查添加当前单词后是否超过长度限制
        if len(current_sentence + " " + word) <= max_length:
            if current_sentence:
                current_sentence += " " + word
            else:
                current_sentence = word
        else:
            # 当前句子已满，保存并开始新句子
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = word
            else:
                # 如果单个单词就超过长度限制，强制分割
                if len(word) > max_length:
                    # 按字符分割，但尽量在合适的位置分割
                    for i in range(0, len(word), max_length):
                        sentences.append(word[i:i + max_length])
                else:
                    current_sentence = word

    # 添加最后一个句子
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def split_by_comma_with_punctuation(text, max_length=50):
    """
    按逗号分割过长的句子，保持标点符号的完整性

    Args:
        text: 要分割的文本
        max_length: 最大句子长度

    Returns:
        sentences: 分割后的句子列表
    """
    if len(text) <= max_length:
        return [text]

    if '，' in text:
        parts = text.split('，')
        sentences = []
        current_sentence = ""

        for i, part in enumerate(parts):
            # 检查当前部分是否包含句子级标点符号
            sentence_punct = ['。', '；', '？', '！', '.', ';', '?', '!']
            has_sentence_punctuation = any(p in part for p in sentence_punct)

            if (len(current_sentence + part + '，') <= max_length and
                    not has_sentence_punctuation):
                if current_sentence:
                    current_sentence += '，' + part
                else:
                    current_sentence = part
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = part
                else:
                    sentences.append(part)

        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    # 如果没有逗号，按长度强制分割
    sentences = []
    for i in range(0, len(text), max_length):
        sentences.append(text[i:i + max_length])

    return sentences


def process_text_with_timestamps(cosyvoice, text, prompt_text,
                                 prompt_speech_16k, stream=False,
                                 max_sentence_length=50):
    """
    处理文本并生成带时间戳的音频

    Args:
        cosyvoice: CosyVoice模型实例
        text: 要合成的文本
        prompt_text: 提示文本
        prompt_speech_16k: 提示音频
        stream: 是否使用流式生成
        max_sentence_length: 最大句子长度，用于文本分割

    Returns:
        full_audio: 完整音频张量
        output_data: 包含时间戳的输出数据
    """
    # 1. 文本分割 - 使用自定义分割函数
    sentences = custom_text_split(text, max_length=max_sentence_length)

    # 2. 音频合成和时间戳记录
    audio_segments = []
    sentence_texts = []

    print(f"文本已分割为 {len(sentences)} 个句子")

    for i, sentence in enumerate(sentences):
        print(f"正在处理第 {i+1}/{len(sentences)} 个句子: {sentence}")

        # 合成当前句子 - 收集所有音频片段并拼接
        sentence_audio_segments = []

        # 添加错误处理和空值检查
        try:
            inference_result = cosyvoice.inference_zero_shot(
                sentence,  # tts_text: 要合成的文本
                prompt_text,  # prompt_text: 提示文本（可为空）
                prompt_speech_16k,  # prompt_speech_16k: 提示音频
                stream=stream,
                text_frontend=True
            )

            # 检查推理结果是否为 None
            if inference_result is None:
                print(f"警告: 句子 '{sentence}' 的推理结果为空，跳过此句子")
                continue

            for audio_result in inference_result:
                if audio_result is None:
                    print(f"警告: 句子 '{sentence}' 的音频结果为空，跳过此片段")
                    continue

                if ('tts_speech' not in audio_result or
                        audio_result['tts_speech'] is None):
                    print(f"警告: 句子 '{sentence}' 的 tts_speech 为空，跳过此片段")
                    continue

                audio_segment = audio_result['tts_speech']
                sentence_audio_segments.append(audio_segment)

        except Exception as e:
            print(f"错误: 处理句子 '{sentence}' 时出现异常: {e}")
            continue

        # 检查是否有有效的音频片段
        if not sentence_audio_segments:
            print(f"警告: 句子 '{sentence}' 没有生成有效的音频，跳过此句子")
            continue

        # 将当前句子的所有音频片段拼接成一个完整的音频
        if len(sentence_audio_segments) > 1:
            sentence_audio = torch.cat(sentence_audio_segments, dim=1)
        else:
            sentence_audio = sentence_audio_segments[0]

        # 保存当前句子的完整音频和文本
        audio_segments.append(sentence_audio)
        sentence_texts.append(sentence)

    # 检查是否有有效的音频段
    if not audio_segments:
        raise ValueError("没有生成任何有效的音频段")

    # 3. 计算精确时间戳
    timestamps = calculate_timestamps(audio_segments, cosyvoice.sample_rate)

    # 打印调试信息
    print("\n调试信息 - 每个句子的音频时长:")
    for i, (sentence, (start_ms, end_ms)) in enumerate(
            zip(sentence_texts, timestamps)):
        duration_ms = end_ms - start_ms
        duration_s = duration_ms / 1000
        start_s = start_ms / 1000
        end_s = end_ms / 1000
        print(f"  句子 {i+1}: {duration_s:.2f}秒 "
              f"({start_s:.2f}s-{end_s:.2f}s) - {sentence}")

    # 4. 拼接完整音频
    if len(audio_segments) > 1:
        full_audio = torch.cat(audio_segments, dim=1)
    else:
        full_audio = audio_segments[0]

    # 5. 生成输出数据
    output_data = {
        "audio_file": str(Path(args.output).name),
        "sample_rate": cosyvoice.sample_rate,
        "total_seconds": timestamps[-1][1] / 1000.0 if timestamps else 0.0,
        "sentences": []
    }

    for i, (text, (start_ms, end_ms)) in enumerate(
            zip(sentence_texts, timestamps)):
        output_data["sentences"].append({
            "text": text,
            "start": start_ms / 1000.0,
            "end": end_ms / 1000.0
        })

    return full_audio, output_data


def format_time_srt(seconds):
    """
    将秒数转换为SRT时间格式 (HH:MM:SS,mmm)

    Args:
        seconds: 秒数（浮点数）

    Returns:
        srt_time: SRT格式的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)

    return (f"{hours:02d}:{minutes:02d}:{secs:02d},"
            f"{millisecs:03d}")


def convert_to_srt(sentences_data):
    """
    将句子数据转换为SRT格式

    Args:
        sentences_data: 包含句子和时间戳的数据

    Returns:
        srt_content: SRT格式的字幕内容
    """
    srt_lines = []

    for i, sentence_info in enumerate(sentences_data["sentences"], 1):
        start_time = format_time_srt(sentence_info["start"])
        end_time = format_time_srt(sentence_info["end"])
        text = sentence_info["text"]

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(f"{text}")
        srt_lines.append("")  # 空行分隔

    return "\n".join(srt_lines)


def save_output_file(output_data, output_filename, output_format):
    """
    根据指定格式保存输出文件

    Args:
        output_data: 输出数据
        output_filename: 输出文件名
        output_format: 输出格式 ('json' 或 'srt')
    """
    if output_format == 'json':
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    elif output_format == 'srt':
        srt_content = convert_to_srt(output_data)
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(srt_content)


def main():
    """主函数"""
    global args
    # 解析命令行参数
    args = parse_arguments()

    try:
        # 如果只是测试句子分割，直接执行测试
        if args.test_sentence_split:
            print("测试句子分割功能...")
            print("=" * 50)

            # 使用 --text 参数的值作为测试用例
            test_cases = [args.text] if args.text else ["测试文本"]

            for i, text in enumerate(test_cases, 1):
                print(f"\n测试用例 {i}: '{text}'")
                sentences = custom_text_split(text, args.max_sentence_length)
                print(f"分割结果: {sentences}")
                print(f"句子数量: {len(sentences)}")

                # 验证分割结果
                if sentences:
                    for j, sentence in enumerate(sentences, 1):
                        is_punct = is_only_punctuation(sentence)
                        print(f"  句子 {j}: '{sentence}' (只包含标点: {is_punct})")
                else:
                    print("  结果为空列表")

            print("\n句子分割测试完成！")
            return

        # 验证输入
        validate_inputs(args)

        print(f"正在加载模型: {args.model_path}")
        print(f"音频文件: {args.audio}")
        print(f"输入文本: {args.text}")
        if args.prompt_text:
            print(f"提示文本: {args.prompt_text}")
        else:
            print("提示文本: (省略)")
        print(f"输出文件: {args.output}")
        print(f"生成次数: {args.iterations}")
        print(f"随机种子: {args.seed}")
        print(f"流式生成: {args.stream}")
        print(f"JIT加载: {args.load_jit}")
        print(f"TensorRT加载: {args.load_trt}")
        print(f"FP16精度: {args.fp16}")
        print("-" * 50)

        # 初始化CosyVoice模型
        cosyvoice = CosyVoice2(
            args.model_path,
            load_jit=False,  # 按照官方示例设置
            load_trt=False,  # 按照官方示例设置
            load_vllm=False,  # 不使用vllm
            fp16=False  # 按照官方示例设置
        )

        # 加载音频文件
        print("正在加载音频文件...")
        prompt_speech_16k = load_wav(args.audio, 16000)

        # 设置随机种子
        set_all_random_seed(args.seed)

        # 生成语音
        print("开始生成语音...")

        # 首先预览文本分割结果
        print("\n文本分割预览:")
        print("-" * 30)
        sentences = custom_text_split(args.text, args.max_sentence_length)
        print(f"输入文本: '{args.text}'")
        print(f"分割为 {len(sentences)} 个句子:")
        for i, sentence in enumerate(sentences, 1):
            is_punct = is_only_punctuation(sentence)
            print(f"  句子 {i}: '{sentence}' (只包含标点: {is_punct})")
        print("-" * 30)

        for i in tqdm(range(args.iterations), desc="生成进度"):
            if args.iterations > 1:
                set_all_random_seed(args.seed + i)

            # 处理文本并生成带时间戳的音频
            full_audio, output_data = process_text_with_timestamps(
                cosyvoice,
                args.text,
                args.prompt_text,
                prompt_speech_16k,
                args.stream,
                args.max_sentence_length
            )

            # 保存音频文件
            output_filename = args.output
            if args.iterations > 1:
                # 如果有多次生成，在文件名中添加序号
                output_path = Path(args.output)
                output_filename = str(
                    output_path.parent /
                    f"{output_path.stem}_iter_{i+1}{output_path.suffix}")

            torchaudio.save(output_filename, full_audio, cosyvoice.sample_rate)
            print(f"已保存音频文件: {output_filename}")

            # 保存时间戳文件
            if args.timestamps_output:
                timestamps_filename = args.timestamps_output
            else:
                # 默认与音频文件同目录同名，根据格式添加扩展名
                output_path = Path(output_filename)
                if args.format == 'srt':
                    timestamps_filename = str(
                        output_path.parent / f"{output_path.stem}.srt")
                else:
                    timestamps_filename = str(
                        output_path.parent / f"{output_path.stem}.json")

            if args.iterations > 1:
                # 如果有多次生成，在文件名中添加序号
                timestamps_path = Path(timestamps_filename)
                timestamps_filename = str(
                    timestamps_path.parent /
                    f"{timestamps_path.stem}_iter_{i+1}"
                    f"{timestamps_path.suffix}")

            # 使用新的保存函数
            save_output_file(output_data, timestamps_filename, args.format)
            print(f"已保存{args.format.upper()}文件: {timestamps_filename}")

            # 打印时间戳信息
            print("\n时间戳信息:")
            for j, sentence_info in enumerate(output_data["sentences"]):
                print(f"  句子 {j+1}: {sentence_info['text']}")
                print(f"    开始时间: {sentence_info['start']:.2f} 秒")
                print(f"    结束时间: {sentence_info['end']:.2f} 秒")

        print("语音生成完成！")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
