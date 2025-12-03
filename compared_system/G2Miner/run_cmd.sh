#!/bin/bash

# 设置超时时间（3小时 = 10800秒）
TIMEOUT=10800

# 第一条命令
echo "开始执行第一条命令: ./bin/sgl_gpu_base ../subgraph-matching/dataset/friendster/label_1/friendster1.bin rectangle"
timeout $TIMEOUT ./bin/sgl_gpu_base ../subgraph-matching/dataset/friendster/label_1/friendster1.bin rectangle
EXIT_CODE1=$?

if [ $EXIT_CODE1 -eq 124 ]; then
    echo "第一条命令执行超时（超过3小时），已被终止"
elif [ $EXIT_CODE1 -ne 0 ]; then
    echo "第一条命令执行失败，退出代码: $EXIT_CODE1"
else
    echo "第一条命令执行成功"
fi

echo "----------------------------------------"

# 第二条命令
echo "开始执行第二条命令: ./bin/sgl_gpu_base ../subgraph-matching/dataset/friendster/label_1/friendster1.bin diamond"
timeout $TIMEOUT ./bin/sgl_gpu_base ../subgraph-matching/dataset/friendster/label_1/friendster1.bin diamond
EXIT_CODE2=$?

if [ $EXIT_CODE2 -eq 124 ]; then
    echo "第二条命令执行超时（超过3小时），已被终止"
elif [ $EXIT_CODE2 -ne 0 ]; then
    echo "第二条命令执行失败，退出代码: $EXIT_CODE2"
else
    echo "第二条命令执行成功"
fi

echo "所有命令执行完毕"