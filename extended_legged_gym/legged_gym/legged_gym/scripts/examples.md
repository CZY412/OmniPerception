# 指定动态链接器查找共享库（.so 文件）的额外路径
export LD_LIBRARY_PATH=/home/hithcat/miniconda3/envs/fangzhen/lib:$LD_LIBRARY_PATH

# 训练
python3 /home/hithcat/Code/tools/PegasusFlow/extended_legged_gym/legged_gym/legged_gym//scripts/train.py --task=anymal_c_flat

# 测试
python3 /home/hithcat/Code/tools/PegasusFlow/extended_legged_gym/legged_gym/legged_gym/scripts/play.py \
        --task anymal_c_flat \
        --num_envs 16 \
        --load_run Nov11_22-44-25_ \
        --checkpoint -1

# 训练参数设置
| 参数                     | 类型   | 说明                       |
| ---------------------- | ---- | ------------------------ |
| `--sim_device`         | str  | 物理仿真设备，`cuda` 或 `cpu`    |
| `--rl_device`          | str  | RL 算法训练设备，`cuda` 或 `cpu` |
| `--graphics_device_id` | int  | 渲染 GPU 编号（用于显示窗口）        |
| `--headless`           | bool | 无界面运行，训练更快               |
| `--num_envs`  | int | 并行环境数量（机器人数量），显存越多可以设置越大 |
| `--subscenes` | int | 可选，渲染拆分场景数量              |
| `--slices`    | int | 可选，物理仿真切片数量              |
| `--max_iterations`  | int  | 总训练迭代次数（相当于原来的 `num_steps`） |
| `--resume`          | bool | 从 checkpoint 恢复训练           |
| `--checkpoint`      | str  | 指定保存 checkpoint 路径          |
| `--experiment_name` | str  | 训练实验名称（用于日志和保存文件夹）          |
| `--run_name`        | str  | 单次运行名称                      |
| `--load_run`        | str  | 加载已有实验用于测试或继续训练             |
| `--num_threads`    | 物理仿真线程数                               |
| `--pipeline`       | 物理仿真管线类型（PhysX GPU pipeline / Flex 等） |
| `--flex / --physx` | 强制选择物理引擎                              |
| `--seed`           | 随机种子                                  |
| `--horovod`        | 分布式训练                                 |

# 训练参数示例
python3 /home/hithcat/Code/tools/PegasusFlow/extended_legged_gym/legged_gym/legged_gym/scripts/train.py \
    --task anymal_c_flat \
    --sim_device cuda \
    --rl_device cuda \
    --graphics_device_id 0 \
    --num_envs 5 \
    --max_iterations 10

