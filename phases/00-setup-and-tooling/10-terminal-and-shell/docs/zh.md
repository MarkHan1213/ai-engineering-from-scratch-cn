# Terminal & Shell / 终端与 Shell

> 终端是 AI 工程师长期生活的地方。你需要在这里变得熟练。

**类型：** 学习
**语言：** --
**前置要求：** Phase 0, Lesson 01
**时间：** 约 35 分钟

## Learning Objectives / 学习目标

- 使用 piping、redirect 和 `grep` 在命令行中过滤和处理训练日志
- 创建持久化 tmux session，并用多个 pane 同时运行训练和 GPU 监控
- 使用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统与 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地与远程机器之间传输文件

## The Problem / 问题

你在终端里花的时间会比任何编辑器都多。训练任务、GPU 监控、日志跟踪、远程 SSH session、环境管理，每个 AI workflow 都会碰到 shell。如果你在这里慢，整个工作都会慢。

这一课只讲 AI 工作真正需要的终端技能。不讲 Unix 历史，不深入 Bash 脚本，只覆盖你马上会用到的东西。

## The Concept / 概念

```mermaid
graph TD
    subgraph tmux["tmux session: training"]
        subgraph top["Top row"]
            P1["Pane 1: Training run<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["Pane 2: GPU monitor<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["Pane 3: Logs + experiments<br/>tail -f logs/train.log | grep loss"]
    end
```

三件事同时运行。一个终端。你可以 detach、回家、重新 SSH 上来并 reattach。训练会继续运行。

## Build It / 动手构建

### Step 1: Know your shell / 第 1 步：了解你的 shell

检查你正在运行哪个 shell：

```bash
echo $SHELL
```

大多数系统使用 `bash` 或 `zsh`。两者都没问题。本课程中的命令在二者中都能工作。

你需要掌握的基础：

```bash
# Move around
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# History search (most useful shortcut you'll learn)
# Ctrl+R then type part of a previous command
# Press Ctrl+R again to cycle through matches

# Clear terminal
clear   # or Ctrl+L

# Cancel a running command
# Ctrl+C

# Suspend a running command (resume with fg)
# Ctrl+Z
```

### Step 2: Piping and redirects / 第 2 步：Piping 和 redirect

Piping 会把命令连接起来。你会用它处理日志、过滤输出、串联工具，而且会非常频繁。

```bash
# Count how many times "loss" appears in a log
cat train.log | grep "loss" | wc -l

# Extract just the loss values from training output
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# Watch a log file update in real time, filtering for errors
tail -f train.log | grep --line-buffered "ERROR"

# Sort experiments by final accuracy
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# Redirect stdout and stderr to separate files
python train.py > output.log 2> errors.log

# Redirect both to the same file
python train.py > train_full.log 2>&1
```

你需要掌握这几个 redirect：

| Symbol | What it does |
|--------|-------------|
| `>` | 把 stdout 写入文件（覆盖） |
| `>>` | 把 stdout 追加到文件 |
| `2>` | 把 stderr 写入文件 |
| `2>&1` | 把 stderr 发送到 stdout 所在的位置 |
| `\|` | 把一个命令的 stdout 作为下一个命令的 stdin |

### Step 3: Background processes / 第 3 步：后台进程

训练通常需要几个小时。你不应该一直开着同一个终端窗口等待。

```bash
# Run in background (output still goes to terminal)
python train.py &

# Run in background, immune to hangup (closing terminal won't kill it)
nohup python train.py > train.log 2>&1 &

# Check what's running in background
jobs
ps aux | grep train.py

# Bring a background job to foreground
fg %1

# Kill a background process
kill %1
# or find its PID and kill that
kill $(pgrep -f "train.py")
```

`&`、`nohup` 和 `screen`/`tmux` 的差别：

| Method | Survives terminal close? | Can reattach? |
|--------|-------------------------|---------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（查看 log file） |
| `screen` / `tmux` | 是 | 是 |

任何超过几分钟的任务，都用 tmux。

### Step 4: tmux / 第 4 步：tmux

tmux 可以创建持久化 terminal session，并在里面放多个 pane。它是管理训练任务最有用的单个工具。

```bash
# Install
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# Start a named session
tmux new -s training

# Split horizontally
# Ctrl+B then "

# Split vertically
# Ctrl+B then %

# Navigate between panes
# Ctrl+B then arrow keys

# Detach (session keeps running)
# Ctrl+B then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill a session
tmux kill-session -t training
```

典型 AI workflow session：

```bash
tmux new -s train

# Pane 1: start training
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " to split, then run GPU monitor
watch -n1 nvidia-smi

# Ctrl+B, % to split vertically, tail the logs
tail -f logs/experiment.log

# Now detach with Ctrl+B, d
# SSH out, go get coffee, come back
# tmux attach -t train
```

### Step 5: Monitoring with htop and nvtop / 第 5 步：用 htop 和 nvtop 监控

```bash
# System processes (better than top)
htop

# GPU processes (if you have NVIDIA GPU)
# Install: sudo apt install nvtop (Ubuntu) or brew install nvtop (macOS)
nvtop

# Quick GPU check without nvtop
nvidia-smi

# Watch GPU usage update every second
watch -n1 nvidia-smi

# See which processes are using the GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

你会用到的 `htop` 快捷键：
- `F6` 或 `>`：按列排序（按 memory 排序可以找 memory leak）
- `F5`：切换 tree view（查看子进程）
- `F9`：kill 进程
- `/`：按进程名搜索

### Step 6: SSH for remote GPU boxes / 第 6 步：连接远程 GPU 机器的 SSH

当你租用云 GPU（Lambda、RunPod、Vast.ai）时，会通过 SSH 连接。

```bash
# Basic connection
ssh user@gpu-box-ip

# With a specific key
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# Copy files to remote
scp model.pt user@gpu-box-ip:~/models/

# Copy files from remote
scp user@gpu-box-ip:~/results/metrics.json ./

# Sync a whole directory (faster for many files)
rsync -avz ./data/ user@gpu-box-ip:~/data/

# Port forward (access remote Jupyter/TensorBoard locally)
ssh -L 8888:localhost:8888 user@gpu-box-ip
# Now open localhost:8888 in your browser

# SSH config for convenience
# Add to ~/.ssh/config:
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# Then just:
# ssh gpu
```

### Step 7: Useful aliases for AI work / 第 7 步：AI 工作常用 alias

把这些加入你的 `~/.bashrc` 或 `~/.zshrc`：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

或者只复制你需要的。关键 alias：

```bash
# GPU status at a glance
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Kill all Python training processes
alias killtraining='pkill -f "python.*train"'

# Quick virtual environment activate
alias ae='source .venv/bin/activate'

# Watch training loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

完整集合见 `code/shell_aliases.sh`。

### Step 8: Common AI terminal patterns / 第 8 步：常见 AI 终端模式

这些会在实际工作中反复出现：

```bash
# Run training, log everything, notify when done
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# Compare two experiment logs side by side
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# Find the largest model files (clean up disk space)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# Download a model from Hugging Face
wget https://huggingface.co/model/resolve/main/model.safetensors

# Untar a dataset
tar xzf dataset.tar.gz -C ./data/

# Count lines in all Python files (see how big your project is)
find . -name "*.py" | xargs wc -l | tail -1

# Check disk space (training data fills disks fast)
df -h
du -sh ./data/*

# Environment variable check before training
env | grep -i cuda
env | grep -i torch
```

## Use It / 应用它

本课程中，各工具会这样进入你的工作流：

| Tool | When you use it |
|------|----------------|
| tmux | 每次训练任务（Phase 3+） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 快速后台任务 |
| `htop` / `nvtop` | 调试慢训练和 OOM 错误 |
| SSH + `rsync` | 在云 GPU 上工作 |
| Piping + redirects | 处理实验结果 |
| Aliases | 节省重复命令的时间 |

## Ship It / 交付它

这一课交付的是一套 AI 终端工作流：tmux 管理长训练任务，SSH/`rsync` 连接远程 GPU，log filtering 和 aliases 则把重复操作变成稳定命令。

## Exercises / 练习

1. 安装 tmux，创建一个带三个 pane 的 session，一个运行 `htop`，一个运行 `watch -n1 date`，第三个运行 Python script。Detach 后再 reattach。
2. 把 `code/shell_aliases.sh` 中的 aliases 加入你的 shell config，并用 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 创建一个假的 training log，然后用 `grep`、`tail` 和 `awk` 只提取 loss 值。
4. 为你能访问的 server 配置一个 SSH config entry（或用 `localhost` 练习语法）。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Shell | “The terminal” | 解释你输入命令的程序，比如 bash、zsh、fish |
| tmux | “Terminal multiplexer” | 让你在一个窗口内运行多个 terminal session，并支持 detach/reattach 的程序 |
| Pipe | “The bar thing” | `\|` operator，会把一个命令的输出作为另一个命令的输入 |
| PID | “Process ID” | 分配给每个运行中进程的唯一数字，用于监控或 kill 它 |
| nohup | “No hangup” | 让命令不受 hangup signal 影响运行，因此关闭终端也不会杀掉它 |
| SSH | “Connecting to the server” | Secure Shell，一种在远程机器上运行命令的加密协议 |
