#!/bin/bash
# msame 推理命令

echo '=== inner_32 ==='
./msame --model "topk_test/om/topk_inner_32.om" \
    --input "topk_test/bins/topk_inner_32_input.bin" \
    --output "topk_test/results/inner_32" \
    --loop 1

echo '=== inner_64 ==='
./msame --model "topk_test/om/topk_inner_64.om" \
    --input "topk_test/bins/topk_inner_64_input.bin" \
    --output "topk_test/results/inner_64" \
    --loop 1

echo '=== inner_256 ==='
./msame --model "topk_test/om/topk_inner_256.om" \
    --input "topk_test/bins/topk_inner_256_input.bin" \
    --output "topk_test/results/inner_256" \
    --loop 1

echo '=== inner_1024 ==='
./msame --model "topk_test/om/topk_inner_1024.om" \
    --input "topk_test/bins/topk_inner_1024_input.bin" \
    --output "topk_test/results/inner_1024" \
    --loop 1

echo '=== inner_4096 ==='
./msame --model "topk_test/om/topk_inner_4096.om" \
    --input "topk_test/bins/topk_inner_4096_input.bin" \
    --output "topk_test/results/inner_4096" \
    --loop 1

echo '=== inner_4128 ==='
./msame --model "topk_test/om/topk_inner_4128.om" \
    --input "topk_test/bins/topk_inner_4128_input.bin" \
    --output "topk_test/results/inner_4128" \
    --loop 1

echo '=== inner_8192 ==='
./msame --model "topk_test/om/topk_inner_8192.om" \
    --input "topk_test/bins/topk_inner_8192_input.bin" \
    --output "topk_test/results/inner_8192" \
    --loop 1

echo '=== inner_32400 ==='
./msame --model "topk_test/om/topk_inner_32400.om" \
    --input "topk_test/bins/topk_inner_32400_input.bin" \
    --output "topk_test/results/inner_32400" \
    --loop 1

echo '=== inner_324000 ==='
./msame --model "topk_test/om/topk_inner_324000.om" \
    --input "topk_test/bins/topk_inner_324000_input.bin" \
    --output "topk_test/results/inner_324000" \
    --loop 1

echo '=== inner_not32 ==='
./msame --model "topk_test/om/topk_inner_not32.om" \
    --input "topk_test/bins/topk_inner_not32_input.bin" \
    --output "topk_test/results/inner_not32" \
    --loop 1

echo '=== outter_4 ==='
./msame --model "topk_test/om/topk_outter_4.om" \
    --input "topk_test/bins/topk_outter_4_input.bin" \
    --output "topk_test/results/outter_4" \
    --loop 1

echo '=== k_equals_n ==='
./msame --model "topk_test/om/topk_k_equals_n.om" \
    --input "topk_test/bins/topk_k_equals_n_input.bin" \
    --output "topk_test/results/k_equals_n" \
    --loop 1

echo '=== k_equals_1 ==='
./msame --model "topk_test/om/topk_k_equals_1.om" \
    --input "topk_test/bins/topk_k_equals_1_input.bin" \
    --output "topk_test/results/k_equals_1" \
    --loop 1
