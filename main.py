#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepResearch Agent 主入口文件

使用方法:
    python main.py --config configs/config_main.yaml --task "你的任务描述"
    
或者直接运行:
    python main.py
    然后输入任务描述
"""

import asyncio
import argparse
import logging
import os
from argparse import Namespace
from typing import Optional

from src.config import config
from src.models import model_manager
from src.logger import logger
from src.orchestrator import TopLevelOrchestrator
async def main():
    """主函数：初始化配置，创建智能体并运行任务"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="DeepResearch Agent - 一个强大的多智能体研究系统"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_main.yaml",
        help="配置文件路径 (默认: configs/config_main.yaml)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="要执行的任务描述"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="最大执行步数 (覆盖配置文件中的设置)"
    )
    
    args = parser.parse_args()
    
    # 初始化配置
    logger.info("=" * 60)
    logger.info("DeepResearch Agent 启动中...")
    logger.info("=" * 60)
    
    # 创建命名空间对象用于配置初始化
    config_args = Namespace(
        config=args.config,
        cfg_options={}
    )
    
    # 如果指定了 max_steps，添加到配置选项
    if args.max_steps:
        config_args.cfg_options['agent_config'] = {
            'max_steps': args.max_steps
        }
    
    # 初始化配置管理器
    config.init_config(args.config, config_args)
    
    # 初始化日志文件输出
    log_path = config.get("log_path", "log.txt")
    logger.init_logger(log_path, level=logging.INFO)
    logger.info(f"| 日志文件路径: {log_path}")
    
    # 初始化模型管理器
    model_manager.init_models()
    logger.info(f"| 已注册 {len(model_manager.registered_models)} 个模型")
    
    logger.info("| 使用顶层 orchestrator 模式")
    orchestrator = TopLevelOrchestrator(config=config)
    
    # 获取任务
    if args.task:
        task = args.task
    else:
        # 交互式输入任务
        print("\n" + "=" * 60)
        print("请输入您的任务（输入 'quit' 或 'exit' 退出）:")
        print("=" * 60)
        task = input("> ").strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            logger.info("| 用户退出")
            return
    
    if not task:
        logger.error("| 错误: 任务描述不能为空")
        return
    
    # 运行智能体
    logger.info("=" * 60)
    logger.info(f"| 开始执行任务: {task}")
    logger.info("=" * 60)
    
    try:
        max_steps = args.max_steps if args.max_steps else None
        
        result = await orchestrator.run(task)
        
        # 显示最终结果
        logger.info("=" * 60)
        logger.info("| 任务执行完成")
        logger.info("=" * 60)
        print("\n" + "=" * 60)
        print("最终结果:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        # 保存结果（如果配置了保存路径）
        if hasattr(config, 'save_path') and config.save_path:
            import json
            import os
            save_path = config.save_path
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            result_data = {
                "task": task,
                "result": str(result),
                "steps": None,
                "token_usage": None
            }
            
            # 追加模式保存（如果是 .jsonl 格式）
            if save_path.endswith('.jsonl'):
                with open(save_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            else:
                # JSON 格式
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"| 结果已保存到: {save_path}")
        
    except KeyboardInterrupt:
        logger.info("\n| 用户中断执行")
    except Exception as e:
        logger.error(f"| 执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    asyncio.run(main())

