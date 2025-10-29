#!/usr/bin/env python3
"""
CosyVoice 语音克隆示例
支持通过命令行参数传入待克隆的音频文件和输入文本
"""

import argparse
import sys
from pathlib import Path

# 添加第三方库路径
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm
import torchaudio


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='CosyVoice 语音克隆示例',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python cosyvoice_clone_example.py --audio ./asset/zero_shot_prompt.wav --text "你好，这是一个语音克隆测试"
  python cosyvoice_clone_example.py --audio ./samples/voice_sample.wav --text "今天天气真不错" --output ./output/cloned_speech.wav
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


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    try:
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
        for i in tqdm(range(args.iterations), desc="生成进度"):
            if args.iterations > 1:
                set_all_random_seed(args.seed + i)
            
            # 执行语音克隆推理
            for j, audio_result in enumerate(cosyvoice.inference_zero_shot(
                args.text,  # tts_text: 要合成的文本
                args.prompt_text,  # prompt_text: 提示文本（可为空）
                prompt_speech_16k,  # prompt_speech_16k: 提示音频
                stream=args.stream,
                text_frontend=False  # 按照官方示例设置
            )):
                # 保存音频文件
                output_filename = args.output
                if args.iterations > 1:
                    # 如果有多次生成，在文件名中添加序号
                    output_path = Path(args.output)
                    output_filename = str(output_path.parent / f"{output_path.stem}_iter_{i+1}{output_path.suffix}")
                
                if j > 0:
                    # 如果有多个音频块，在文件名中添加块序号
                    output_path = Path(output_filename)
                    output_filename = str(output_path.parent / f"{output_path.stem}_chunk_{j+1}{output_path.suffix}")
                
                torchaudio.save(output_filename, audio_result['tts_speech'], cosyvoice.sample_rate)
                print(f"已保存音频文件: {output_filename}")
        
        print(f"语音生成完成！")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
